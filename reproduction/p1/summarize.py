"""Recompute S0 score metrics, account for model buffers, and freeze P2 budgets."""
import csv,gc,hashlib,json,math,pickle,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from genimage_detector.metrics import binary_metrics
P=ROOT/'reproduction/p1';C=json.loads((P/'protocol.json').read_text());E=json.loads((P/'extended_protocol.json').read_text());LOG=Path(C['output_root'])
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def main():
 methods=C['methods']+E['methods'];rows=[];checks=[]
 deltas={r['method']:r for r in json.loads((P/'delta_validation.json').read_text())['methods']}
 with open(C['validation_manifest']) as f:expected=list(csv.DictReader(f))
 for method in methods:
  p=LOG/method;r=json.loads((p/'result.json').read_text());assert r['status']=='passed' and r['test_access'] is False
  config=P/('protocol.json' if method in C['methods'] else 'extended_protocol.json');assert r['protocol_sha256']==sha(config)
  assert r['train_originals']==4000 and r['validation_originals']==1024;assert r['train_cost']['measured_batches']>=100;assert all(r['checks'].values())
  assert sha(r['checkpoint'])==r['checkpoint_sha256']
  with (p/'validation_scores.csv').open() as f:scores=list(csv.DictReader(f))
  assert [(x['path'],x['original_id'],x['label']) for x in scores]==[(x['path'],x['original_id'],x['label']) for x in expected]
  y=np.array([int(x['label']) for x in scores]);prob=np.array([float(x['score']) for x in scores]);th=json.loads((p/'thresholds.json').read_text())['thresholds']
  for name,t in th.items():
   m=binary_metrics(y,prob,t)
   for k,v in m.items():
    if v is not None:assert np.isclose(v,r['metrics'][name][k],rtol=1e-10,atol=1e-12),(method,name,k)
   assert [int(x['prediction_'+name]) for x in scores]==(prob>=t).astype(int).tolist(),(method,name,'prediction mismatch')
  assert r['metrics']['FPR01']['real_fpr']<=.01
  if r['kind']=='frozen_features_logistic':
   clf=pickle.loads(Path(r['checkpoint']).read_bytes());learned=clf.coef_.size+clf.intercept_.size;floating=r['total_parameters']+learned;optimized=learned
  else:
   st=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);floating=sum(v.numel() for v in st['model'].values() if v.is_floating_point());learned=r['trainable_parameters'];param_shapes=json.loads((p/'trainable_parameters.json').read_text());ids=[i for group in st['optimizer']['param_groups'] for i in group['params']];assert len(ids)==len(param_shapes)
   optimized=sum(math.prod(shape) for pid,shape in zip(ids,param_shapes.values()) if pid in st['optimizer']['state']);del st;gc.collect()
  delta_bytes=r['checkpoint_bytes'] if method not in deltas else deltas[method]['bytes']
  if method in deltas:assert sha(deltas[method]['file'])==deltas[method]['sha256']
  row=dict(delta_checkpoint_GiB=delta_bytes/2**30,method=method,train_images_per_second=r['train_cost']['images_per_second'],measured_batches=r['train_cost']['measured_batches'],seconds_per_update=r['train_cost'].get('seconds_per_update'),validation_images_per_second=r['validation_cost']['images_per_second'],peak_allocated_GiB=r['train_cost']['peak_allocated_GiB'],checkpoint_GiB=r['checkpoint_bytes']/2**30,registered_model_parameters=r['total_parameters'],floating_model_state_elements=floating,learned_parameters=learned,parameters_with_optimizer_state=optimized,initialization_seconds=r['initialization_seconds'],wall_seconds=r['total_wall_seconds'],fit_seconds=r.get('classifier_fit_seconds'),precision=r.get('precision','FP32 features / CPU logistic'),source_result=str(p/'result.json'))
  rows.append(row);checks.append(dict(method=method,status='passed',score_rows=len(scores),checkpoint_sha256=r['checkpoint_sha256']))
 tta=json.loads((LOG/'iapl_tta/result.json').read_text());assert tta['status']=='passed' and all(tta['checks'].values()) and not tta['test_access']
 assert tta['originals_by_source']=={'Real':16,'SD14':8,'BigGAN':8}
 with (LOG/'iapl_tta/validation_probe_scores.csv').open() as f:tta_rows=list(csv.DictReader(f))
 assert len(tta_rows)==64 and len({(r['original_id'],r['condition']) for r in tta_rows})==64
 assert {r['original_id'] for r in tta_rows}<={r['original_id'] for r in expected}
 # All recipes are predeclared before final test access. This writes configs only.
 queue=[]
 upstreams={r['name']:r for r in json.loads((ROOT/'reproduction/patches/registry.json').read_text())}
 native_sampling=['effort_native','lora_native','effort_npr','npr_only','effort_rgb','effort_npr_noaux','resnet50','deit_s','swin_t','cnnspot']
 for row in rows:
  method=row['method'];frozen=method.startswith('frozen_') and method!='frozen_only'
  for scale,n in [('S1',16000),('S2',64000)]:
   for seed in [42,43,44]:
    epochs=None if frozen else 1 if method=='iapl' else 5
    gpu_sec=(n/row['train_images_per_second']+18000/row['validation_images_per_second']) if frozen else epochs*(n/row['train_images_per_second']+10000/row['validation_images_per_second'])+8000/row['validation_images_per_second']
    gpu_sec+=row['initialization_seconds']
    # TTA validation+dev is additional, not hidden in the plain-forward number.
    tta_hours=(18000*tta['mean_seconds_per_image']/3600) if method=='iapl' else 0.
    q=dict(sampler='class_balanced_with_replacement' if method in native_sampling else 'fixed_seed_random_permutation',augmentation=method,model_source='P0 fixed upstream plus P1 documented bindings',id=f'U_{method}_{scale}_s{seed}',track='U',method=method,scale=scale,seed=seed,train_originals=n,epochs=epochs,feature_passes=1 if frozen else None,batch_size=32 if frozen else 16,gradient_accumulation=1,updates=None if frozen else epochs*math.ceil(n/16),optimizer='sklearn_LogisticRegression_C1_balanced_maxiter3000' if frozen else ('AdamW' if method.startswith('dual_') or method in ['frozen_only','adapter_only','ssf_only'] else 'Adam'),learning_rate=None if frozen else 5e-5 if method=='iapl' else 1e-4 if method in ['lora_native','resnet50','deit_s','swin_t','cnnspot'] else 2e-4,weight_decay=None if frozen else 0. if method=='iapl' or method.startswith('dual_') or method in ['frozen_only','adapter_only','ssf_only'] else 5e-4,precision=row['precision'],checkpoint_selection='single fit' if frozen else 'max validation AUROC (plain inference); no test selection',threshold_rule='0.5/max BA/FPR<=1% fitted on full 10000 validation after checkpoint selection; same pipeline for TTA',validation_n=10000,dev_n=8000,estimated_gpu_hours_without_retry=gpu_sec/3600,estimated_extra_TTA_validation_dev_hours=tta_hours,estimated_gpu_hours_with_20pct_retry=(gpu_sec/3600+tta_hours)*1.2,checkpoint_format='verified_delta_v1' if not frozen else 'sklearn_pickle_with_encoder_revision',checkpoint_storage_GiB=2*row['delta_checkpoint_GiB'],classifier_CPU_fit_estimate_seconds=None if not frozen else row['fit_seconds']*(n/4000)**2,status='frozen_not_launched',cost_basis=row['source_result'],final_test_access=False)
    upstream='ARIN5201_CV_Project' if frozen else 'IAPL' if method=='iapl' else 'clip_dual_aigi' if method.startswith('dual_') or method in ['frozen_only','adapter_only','ssf_only'] else 'ARIN5201-AIGI-Detection'
    q.update(train_manifest=str(ROOT/'reproduction/manifests/v1_20261114'/('train_'+scale+'.csv')),train_manifest_sha256=sha(ROOT/'reproduction/manifests/v1_20261114'/('train_'+scale+'.csv')),upstream=upstreams[upstream],environment_lock_sha256=sha(ROOT/'reproduction/envs/core.lock'),weights_lock_sha256=sha(ROOT/'reproduction/weights.lock.json'),extra_weights_lock_sha256=sha(P/'extra_weights.lock.json'),method_code_sha256={f:sha(P/f) for f in ['models.py','models_extended.py','checkpoint.py']})
    queue.append(q)
 # Fixed-update comparisons supplement the fixed-epoch scale curves.
 for method in ['effort','effort_degradation']:
  for seed in [42,43,44]:
   base=next(x for x in queue if x['method']==method and x['scale']=='S2' and x['seed']==seed);q=dict(base,id=f'U_{method}_S2_fixed5000updates_s{seed}',epochs=None,updates=5000,sample_presentations=80000,stop_rule='exactly 5000 optimizer updates; preserve partial-epoch sampler cursor')
   rate=next(r['train_images_per_second'] for r in rows if r['method']==method);v=next(r['validation_images_per_second'] for r in rows if r['method']==method);q['estimated_gpu_hours_without_retry']=(80000/rate+2*10000/v+8000/v)/3600;q['estimated_gpu_hours_with_20pct_retry']=q['estimated_gpu_hours_without_retry']*1.2;queue.append(q)
 # Frozen models have no neural update count: fixed solver max_iter/C/tol plus fit CPU accounting.
 budget=dict(status='frozen_not_launched',row_count=len(queue),core_rows=len(rows)*6,fixed_update_extra_rows=6,gpu_hours_with_20pct_retry=sum(q['estimated_gpu_hours_with_20pct_retry'] for q in queue),checkpoint_storage_GiB_with_20pct_retry=sum(q['checkpoint_storage_GiB'] for q in queue)*1.2,feature_and_score_reserve_GiB=20,estimation_limits=['S0 throughput extrapolation, not measured S1/S2 cost','Five epochs fixed for neural methods except native one-epoch IAPL; best checkpoint chosen using full validation, no early stopping','Single-seed throughput, desktop GPU display load present','Logistic CPU fit budget uses conservative quadratic extrapolation; not GPU time','No pure data-scaling causal claim from fixed epochs','TTA validation and dev costs included; P3 final ten-condition panel cost is separate'],IAPL_final_19000x10_seconds_per_checkpoint=190000*tta['mean_seconds_per_image'],IAPL_final_19000x10_hours_per_checkpoint=190000*tta['mean_seconds_per_image']/3600,recipes=queue)
 dump(P/'s1_s2_queue.json',budget)
 with (P/'throughput.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 dump(P/'results.json',dict(status='passed',methods=rows,IAPL_TTA=tta,budget={k:v for k,v in budget.items() if k!='recipes'},metric_scope='All S0 validation metrics diagnostic only; never use as final performance ranking'))
 dump(P/'verification.json',dict(status='passed',methods=checks,score_rows_verified=len(methods)*1024,TTA_checks=tta['checks'],test_access=False))
 print('Verified',len(rows),'methods;',len(queue),'frozen candidate runs;',round(budget['gpu_hours_with_20pct_retry'],2),'GPU hours including TTA validation/dev +20%')
if __name__=='__main__':main()
