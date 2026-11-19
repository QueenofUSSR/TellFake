"""Recompute all final metrics, deduplicate shared reals, paired cluster bootstrap."""
import json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from genimage_detector.data import read_csv,write_csv,write_json,file_hash,load_manifests,load_supplement
from genimage_detector.metrics import binary_metrics,grouped_metrics
ROOT=Path('/mnt/ssd/data/logs/arin5201');FINAL=ROOT/'final_20261113'
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/ssd/code/arin5201/results/final_20261113')
GENS=['BigGAN','SD15','VQDM','Midjourney'];METHODS=['A','B','C']

def main():
 runs=json.loads((FINAL/'verified_runs.json').read_text());protocol=json.loads((FINAL/'evaluation_protocol.json').read_text());conds=protocol['conditions']
 base,fp=load_manifests('data/manifests/v3');supp,meta,sfp=load_supplement('data/manifests/v3_supplement_20261113',base,fp)
 OUT.mkdir(parents=True,exist_ok=False)
 expected={n:{r['path'] for r in base[n]+base['real_test']} for n in ['seen_test','unseen_test']}
 expected['midjourney_test']={r['path'] for r in supp['midjourney_test']}|{'GenImage/official_v3/'+r['path'] for r in base['real_test']}
 expected['external_real_test']={r['path'] for r in supp['external_real_test']}
 data={};metrics=[];ledger=[];training=[]
 def canonical(path):return path if path.startswith(('GenImage/','val2017/')) else 'GenImage/official_v3/'+path
 reference_rows={canonical(r['path']):r for rs in list(base.values())+list(supp.values()) for r in rs}
 for run in runs:
  name=run['name'];method=run['method'];seed=run['seed'];directory=ROOT/run['evaluation'];report=json.loads((directory/'metrics.json').read_text())
  config=json.loads((ROOT/name/'config.json').read_text())
  assert config['epochs']==10 and config['batch_size']==64 and config['seed']==seed and config['arch']=='resnet50'
  assert config.get('method','A')==method and config.get('consistency_weight',0)==(.1 if method=='C' else 0)
  assert config['target_fpr']==.01 and config['lr']==.0001 and config['weight_decay']==.0001
  assert file_hash(ROOT/name/'train_used.csv')==fp['train'] and file_hash(ROOT/name/'val_used.csv')==fp['val']

  assert report['checkpoint_sha256']==run['checkpoint_sha256'] and report['threshold']==run['threshold'] and not report['smoke_test']
  assert report['manifest_sha256']==fp and report['supplement_manifest_sha256']==sfp
  for condition in conds:
   real_reference=None
   for split in expected:
    path=directory/f"{split}_{condition.replace(':','_')}.csv";rows=read_csv(path)
    assert len(rows)==len(expected[split]) and {r['path'] for r in rows}==expected[split]
    assert len({r['path'] for r in rows})==len(rows)
    assert all(int(r['label'])==int(reference_rows[canonical(r['path'])]['label']) and r['generator']==reference_rows[canonical(r['path'])]['generator'] for r in rows)
    y=[int(r['label']) for r in rows];p=[float(r['p_ai']) for r in rows]
    assert all(int(r['prediction'])==int(float(r['p_ai'])>=run['threshold']) for r in rows)
    if split!='external_real_test':
     rec=grouped_metrics(y,p,[r['generator'] for r in rows],run['threshold']);assert rec==report['results'][split][condition]
    else:
     rec=binary_metrics(y,p,run['threshold']);assert rec['real_fpr']==report['results'][split][condition]['overall']['real_fpr']
    ledger.append(dict(run=name,split=split,condition=condition,path=str(path),sha256=file_hash(path),n=len(rows)))
    grouped=defaultdict(list)
    for row in rows:grouped[row['generator']].append(row)
    for gen,rs in grouped.items():
     keygen='COCO' if split=='external_real_test' else ('ImageNet' if gen=='Real' else gen)
     rs=sorted(rs,key=lambda r:canonical(r['path']))
     values=(tuple(canonical(r['path']) for r in rs),np.array([float(r['p_ai']) for r in rs]),np.array([int(r['prediction']) for r in rs],dtype=float))
     key=(method,seed,condition,keygen)
     if key in data:
      assert data[key][0]==values[0];np.testing.assert_array_equal(data[key][1],values[1])
     else:data[key]=values
   real=data[(method,seed,condition,'ImageNet')][1]
   for gen in GENS:
    scores=data[(method,seed,condition,gen)][1]
    m=binary_metrics([0]*len(real)+[1]*len(scores),np.concatenate([real,scores]),run['threshold'])
    metrics.append(dict(method=method,seed=seed,condition=condition,group=gen,**m))
   for gen in ['ImageNet','COCO']:
    scores=data[(method,seed,condition,gen)][1];m=binary_metrics([0]*len(scores),scores,run['threshold'])
    m.update(balanced_accuracy=None,precision=None,recall=None,f1=None)
    metrics.append(dict(method=method,seed=seed,condition=condition,group=gen,**m))
  history=[json.loads(l) for l in (ROOT/name/'history.jsonl').read_text().splitlines()]
  for h in history:training.append(dict(method=method,seed=seed,epoch=h['epoch'],train_loss=h['train_loss'],val_loss=h['val_loss'],seconds=h['seconds'],amp_retries=h.get('amp_batch_retries',0)))
 write_csv(OUT/'per_seed_metrics.csv',metrics,tuple(metrics[0]));write_csv(OUT/'prediction_ledger.csv',ledger,tuple(ledger[0]));write_csv(OUT/'training.csv',training,tuple(training[0]))
 keys=['auroc','balanced_accuracy','recall','real_fpr'];summary=[]
 for method in METHODS:
  for condition in conds:
   for gen in GENS+['ImageNet','COCO']:
    group=[r for r in metrics if r['method']==method and r['condition']==condition and r['group']==gen]
    for metric in keys:
     vals=[r[metric] for r in group if r[metric] is not None]
     if not vals:continue
     summary.append(dict(method=method,condition=condition,group=gen,metric=metric,mean=float(np.mean(vals)),std=float(np.std(vals,ddof=1)),seeds=3))
 write_csv(OUT/'seed_summary.csv',summary,tuple(summary[0]))
 # Mean across conditions computed per seed first. Shared ImageNet FPR counted once.
 overall=[]
 for scope,indices in [('clean',[0]),('non_strong',list(range(7))),('strong',[7]),('all',list(range(8)))]:
  for method in METHODS:
   for metric in ['macro_auroc','macro_balanced_accuracy','macro_recall','unknown_recall','ImageNet_FPR','COCO_FPR','worst_group_balanced_accuracy']:
    seedvals=[]
    for seed in [42,43,44]:
     rs=[r for r in metrics if r['method']==method and r['seed']==seed and r['condition'] in [conds[i] for i in indices]]
     if metric.startswith('macro_'):vals=[r[metric[6:]] for r in rs if r['group'] in GENS]
     elif metric=='unknown_recall':vals=[r['recall'] for r in rs if r['group'] in ['VQDM','Midjourney']]
     elif metric.endswith('_FPR'):vals=[r['real_fpr'] for r in rs if r['group']==metric[:-4]]
     else:vals=[min(r['balanced_accuracy'] for r in rs if r['group'] in GENS)]
     seedvals.append(float(np.mean(vals)))
    overall.append(dict(scope=scope,method=method,metric=metric,mean=float(np.mean(seedvals)),std=float(np.std(seedvals,ddof=1)),seed42=seedvals[0],seed43=seedvals[1],seed44=seedvals[2]))
 write_csv(OUT/'overall_summary.csv',overall,tuple(overall[0]))
 paired_seeds=[]
 for c in [r for r in overall if r['method']=='C']:
  b=next(r for r in overall if r['method']=='B' and r['scope']==c['scope'] and r['metric']==c['metric'])
  values=[c['seed'+str(seed)]-b['seed'+str(seed)] for seed in [42,43,44]]
  paired_seeds.append(dict(scope=c['scope'],metric=c['metric'],seed42=values[0],seed43=values[1],seed44=values[2],mean=float(np.mean(values)),std=float(np.std(values,ddof=1))))
 write_csv(OUT/'paired_seed_deltas.csv',paired_seeds,tuple(paired_seeds[0]))

 # Paired stratified bootstrap: each row is one original with its 8 conditions.
 rng=np.random.default_rng(protocol['bootstrap_seed']);boots={};points={};reps=protocol['bootstrap_replicates']
 for gen in GENS+['ImageNet','COCO']:
  matrices=[]
  for seed in [42,43,44]:
   columns=[]
   for condition in conds:
    b=data[('B',seed,condition,gen)];c=data[('C',seed,condition,gen)]
    assert b[0]==c[0];columns.append(c[2]-b[2])
   matrices.append(np.stack(columns,axis=1))
  matrix=np.mean(matrices,axis=0);n=len(matrix);points[gen]=matrix.mean(0)
  bootstrap=np.empty((reps,len(conds)))
  for start in range(0,reps,50):
   weights=rng.multinomial(n,np.full(n,1/n),size=min(50,reps-start));bootstrap[start:start+len(weights)]=weights@matrix/n
  boots[gen]=bootstrap
 delta={'macro_recall':(np.mean([points[g] for g in GENS],axis=0),np.mean([boots[g] for g in GENS],axis=0)),
        'unknown_recall':(np.mean([points[g] for g in ['VQDM','Midjourney']],axis=0),np.mean([boots[g] for g in ['VQDM','Midjourney']],axis=0)),
        'ImageNet_FPR':(points['ImageNet'],boots['ImageNet']),'COCO_FPR':(points['COCO'],boots['COCO'])}
 delta['macro_balanced_accuracy']=((delta['macro_recall'][0]-points['ImageNet'])/2,(delta['macro_recall'][1]-boots['ImageNet'])/2)
 intervals=[]
 for metric,(point,boot) in delta.items():
  for scope,idx in [(c,[i]) for i,c in enumerate(conds)]+[('non_strong',list(range(7))),('all',list(range(8)))]:
   value=float(point[idx].mean());lo,hi=np.quantile(boot[:,idx].mean(1),[.025,.975])
   intervals.append(dict(comparison='C_minus_B',scope=scope,metric=metric,delta=value,low95=float(lo),high95=float(hi),replicates=reps))
 for entry in intervals:
  if entry['scope'] not in ('non_strong','all'):continue
  a=next(r for r in overall if r['method']=='C' and r['scope']==entry['scope'] and r['metric']==entry['metric'])
  b=next(r for r in overall if r['method']=='B' and r['scope']==entry['scope'] and r['metric']==entry['metric'])
  assert abs(entry['delta']-(a['mean']-b['mean']))<1e-12
 write_csv(OUT/'paired_cluster_intervals.csv',intervals,tuple(intervals[0]))
 result=dict(runs=runs,conditions=conds,unique_final_images=15995,ImageNet_real=5000,COCO_real=4995,ai_counts=dict(BigGAN=1000,SD15=1000,VQDM=2000,Midjourney=2000),prediction_rows_verified=sum(r['n'] for r in ledger),prediction_files=len(ledger),all_metrics_recomputed=True,shared_real_predictions_identical=True,overall=overall,paired_intervals=intervals,paired_seed_deltas=paired_seeds,limitations=['Confidence intervals conditional on these 3 models; seeds averaged before original-cluster resampling; seed SD reported separately.','Pointwise 95% percentile bootstrap, no multiple-comparison adjustment.','Full unknown-generator generalization beyond VQDM/Midjourney not established.','JPEG/PNG/resolution source bias, possible pretraining overlap, conventional COCO real proxy and imperfect near-duplicate screening.','One B44 failed attempt retained; restarted B44/C44 use documented same-batch AMP replay.'])
 write_json(OUT/'results.json',result)
 write_json(OUT/'verification.json',dict(status='passed',prediction_files=len(ledger),rows=sum(r['n'] for r in ledger),unique_images=15995,seeds=[42,43,44],base_fingerprints=fp,supplement_fingerprints=sfp,final_run_count=9))
 print('Verified final results',len(ledger),'files',sum(r['n'] for r in ledger),'rows',flush=True)

if __name__=='__main__':main()
