"""One real-data S0 run, bounded to train/validation. Atomic completion receipts."""
import argparse,contextlib,csv,gc,hashlib,json,os,pickle,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from reproduction.protocol import read_manifest,assert_disjoint
from reproduction.p1 import models as binding
from genimage_detector.metrics import binary_metrics,choose_threshold,choose_fpr_threshold
P=ROOT/'reproduction/p1';CFG=json.loads((P/'protocol.json').read_text())

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def sync():torch.cuda.synchronize()
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])
def subset(rows,n):return [r for label,k in [(0,n//2),(1,n//2)] for r in [x for x in rows if x['label']==label][:k]]
def read_data():
 assert sha(CFG['train_manifest'])==CFG['train_sha256'];assert sha(CFG['validation_manifest'])==CFG['validation_sha256']
 train=read_manifest(CFG['train_manifest'])
 with open(CFG['validation_manifest']) as f:val=list(csv.DictReader(f))
 for r in val:r['label']=int(r['label'])
 assert_disjoint(train,val);assert all(r['source_split']=='train' for r in train+val)
 return train,val

def loader(rows,tf,batch,schedule=None):
 ds=binding.Images(rows,tf,CFG['seed']);return torch.utils.data.DataLoader(ds,batch_size=batch,sampler=schedule,shuffle=False,num_workers=CFG['num_workers'],pin_memory=True,persistent_workers=False,generator=torch.Generator().manual_seed(CFG['seed']))

def metrics_and_scores(out,rows,prob):
 labels=np.array([r['label'] for r in rows]);thresholds={'p05':.5,'max_BA':choose_threshold(labels,prob),'FPR01':choose_fpr_threshold(labels,prob)}
 metrics={k:binary_metrics(labels,prob,v) for k,v in thresholds.items()}
 with (out/'validation_scores.csv').open('w',newline='') as f:
  fields=['path','original_id','source','condition','label','score','prediction_p05','prediction_max_BA','prediction_FPR01'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r,p in zip(rows,prob):w.writerow(dict(path=r['path'],original_id=r['original_id'],source=r['generator'],condition='clean',label=r['label'],score=float(p),**{'prediction_'+k:int(float(p)>=float(v)) for k,v in thresholds.items()}))
 write(out/'thresholds.json',dict(scope='S0 validation diagnostic; never deploy as formal thresholds',thresholds=thresholds,validation_sha256=CFG['validation_sha256']))
 return metrics

def extract(model,dl,method,measure):
 feats=[];labels=[];times=[];batches=0;ups=0;model.eval();torch.cuda.reset_peak_memory_stats();it=iter(dl)
 with torch.no_grad():
  while True:
   sync();start=time.perf_counter()
   try:x,y,idx=next(it)
   except StopIteration:break
   x=x.cuda(non_blocking=True);kwargs={'interpolate_pos_encoding':True} if 'crop128' in method else {}
   f=model(pixel_values=x,**kwargs).image_embeds;f=torch.nn.functional.normalize(f,dim=-1).float().cpu();assert torch.isfinite(f).all();sync();elapsed=time.perf_counter()-start
   feats.append(f.numpy());labels.extend(y.tolist())
   if batches>=CFG['warmup_batches'] or not measure:times.append((len(y),elapsed))
   batches+=1
   if batches%50==0:print('extract',batches,flush=True)
 assert not measure or len(times)>=CFG['min_measured_batches']
 return np.concatenate(feats),np.array(labels),dict(measured_batches=len(times),images=sum(n for n,t in times),seconds=sum(t for n,t in times),images_per_second=sum(n for n,t in times)/sum(t for n,t in times),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)

def frozen_run(method,model,meta,train,val,out,ckpt):
 trtf=binding.Transform(method,True,meta);vatf=binding.Transform(method,False,meta);bs=CFG['frozen_batch_size']
 X,y,cost=extract(model,loader(train,trtf,bs),method,True);V,vy,vcost=extract(model,loader(val,vatf,bs),method,False)
 key=hashlib.sha256(json.dumps(dict(model_path=meta['path'],train=CFG['train_sha256'],val=CFG['validation_sha256'],method=method,seed=CFG['seed'],implementation=sha(P/'models.py')),sort_keys=True).encode()).hexdigest()
 cache=Path('/mnt/ssd/data/cache/arin5201/reproduction/features')/key;cache.mkdir(parents=True,exist_ok=True);np.savez(cache/'features.npz',train=X,train_labels=y,val=V,val_labels=vy)
 start=time.perf_counter();clf=LogisticRegression(max_iter=3000,C=1.,class_weight='balanced').fit(X,y);fit_sec=time.perf_counter()-start;prob=clf.predict_proba(V)[:,1]
 path=ckpt/'classifier.pkl';path.write_bytes(pickle.dumps(clf));back=pickle.loads(path.read_bytes());np.testing.assert_array_equal(prob,back.predict_proba(V)[:,1])
 # Native extractor equivalence on the identical validation images, no random crop.
 from types import SimpleNamespace
 native=binding.module('p1_thomas_verify',binding.SRC/'ARIN5201_CV_Project/scripts/train_baseline.py')
 args=SimpleNamespace(preprocess='crop' if 'crop128' in method else 'resize',crop_size=128,seed=42,train_crops=1,train_jpeg=None,train_jpeg_p=.5,batch=bs)
 native_X,_,_=native.extract([(r['path'],r['label'],r['generator']) for r in val[:bs]],'/',model,vatf.proc,args,'cuda',False)
 np.testing.assert_allclose(native_X,V[:bs],rtol=2e-4,atol=2e-5)
 return dict(kind='frozen_features_logistic',train_cost=cost,validation_cost=vcost,classifier_fit_seconds=fit_sec,feature_cache=str(cache),feature_cache_bytes=(cache/'features.npz').stat().st_size,checkpoint=str(path),checkpoint_bytes=path.stat().st_size,checkpoint_sha256=sha(path),metrics=metrics_and_scores(out,val,prob),checks=dict(native_eval_features_match=True,checkpoint_predictions_equal=True,encoder_frozen=not any(p.requires_grad for p in model.parameters()),features_finite=True),originals_presented=len(train),updates=None,effective_batch=bs)

def make_checkpoint(model,opt,scheduler,scaler,step,schedule):
 return dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),next_batch=step,sampler=dict(schedule=schedule,epoch=0,next_batch=step),protocol_sha256=sha(P/'protocol.json'),train_sha256=CFG['train_sha256'],validation_sha256=CFG['validation_sha256'])
def restore(model,opt,scheduler,scaler,st):
 assert st['protocol_sha256']==sha(P/'protocol.json');model.load_state_dict(st['model'],strict=True);opt.load_state_dict(st['optimizer']);scheduler.load_state_dict(st['scheduler']);scaler.load_state_dict(st['scaler']);restore_rng(st['rng'])

def train_run(method,model,meta,train,val,out,ckpt):
 bs=CFG['batch_size'];seed=CFG['seed'];gen=torch.Generator().manual_seed(seed)
 schedule=torch.multinomial(torch.ones(len(train)),len(train),replacement=True,generator=gen).tolist() if method.endswith('_native') else torch.randperm(len(train),generator=gen).tolist()
 trtf=binding.Transform(method,True,meta);vatf=binding.Transform(method,False,meta)
 parameters=[p for p in model.parameters() if p.requires_grad];names=[n for n,p in model.named_parameters() if p.requires_grad]
 assert parameters
 if method.startswith('effort'):assert all(any(x in n for x in ['U_residual','S_residual','V_residual','head.']) for n in names)
 if method.startswith('lora'):assert all(any(x in n for x in ['.A','.B','head.']) for n in names)
 if method.startswith('dual_'):assert all(any(x in n for x in ['adapter_','ssf_','logit_','head.']) for n in names)
 if method.endswith('_only'):assert all(any(x in n for x in ['adapter_','ssf_','head.']) for n in names)
 write(out/'trainable_parameters.json',{n:list(p.shape) for n,p in model.named_parameters() if p.requires_grad})
 opt=getattr(torch.optim,meta['optimizer'])(parameters,lr=meta['lr'],weight_decay=meta['weight_decay']);scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=.1);scaler=torch.amp.GradScaler('cuda',enabled=False)
 train_loader=loader(train,trtf,bs,schedule);firstx,firsty,_=next(iter(train_loader));firstx,firsty=firstx[:2].cuda(),firsty[:2].cuda()
 # Keep preflight from changing the actual run's RNG, parameters, or mutable buffers.
 initial=make_checkpoint(model,opt,scheduler,scaler,0,schedule);tmp=ckpt/'preflight.pt';torch.save(initial,tmp);del initial
 checks={};model.train()
 if hasattr(model,'frozen'):assert not model.frozen.training
 for dtype in [None,torch.bfloat16]:
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=dtype or torch.float32,enabled=dtype is not None):l=binding.loss(model,firstx,firsty,meta['kind'])
  assert torch.isfinite(l);l.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
  assert any(p.grad is not None and p.grad.abs().sum()>0 for p in parameters)
  checks['FP32_finite' if dtype is None else 'BF16_finite']=True
 state=torch.load(tmp,weights_only=False,map_location='cpu');restore(model,opt,scheduler,scaler,state);del state
 # Verify a real update is identical after disk checkpoint + optimizer/RNG replay.
 def step_once(x,y):
  model.train();opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=meta['amp']):loss=binding.loss(model,x,y,meta['kind'])
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(parameters,float('inf'),error_if_nonfinite=True);opt.step();return float(loss.detach())
 l1=step_once(firstx,firsty);expected={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
 st=torch.load(tmp,weights_only=False,map_location='cpu');restore(model,opt,scheduler,scaler,st);del st;l2=step_once(firstx,firsty)
 assert l1==l2,(l1,l2)
 for n,p in model.named_parameters():
  if p.requires_grad:torch.testing.assert_close(p.cpu(),expected[n],rtol=1e-5,atol=1e-7)
 del expected
 st=torch.load(tmp,weights_only=False,map_location='cpu');restore(model,opt,scheduler,scaler,st);del st;tmp.unlink();checks['disk_optimizer_rng_step_replay']=True
 # Verify repeated train/eval toggles cannot unfreeze the frozen branch.
 model.eval();model.train()
 if hasattr(model,'frozen'):assert not model.frozen.training
 checks['train_eval_switch']=True
 times=[];losses=[];start_run=time.perf_counter();torch.cuda.reset_peak_memory_stats();it=iter(train_loader);batch=0;allpresented=[]
 while True:
  sync();start=time.perf_counter()
  try:x,y,idx=next(it)
  except StopIteration:break
  x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);value=step_once(x,y);sync();sec=time.perf_counter()-start
  if batch>=CFG['warmup_batches']:times.append((len(y),sec))
  losses.append(value);allpresented.extend(idx.tolist());batch+=1
  if batch%25==0:print('train',method,batch,'/',len(train_loader),'loss',round(value,5),'sec',round(time.perf_counter()-start_run),flush=True)
  if batch==len(train_loader)//2:
   st=make_checkpoint(model,opt,scheduler,scaler,batch,schedule);torch.save(st,ckpt/'last.pt');del st
 assert len(times)>=CFG['min_measured_batches'];scheduler.step();trainsec=time.perf_counter()-start_run;peak=torch.cuda.max_memory_allocated()/2**30;reserved=torch.cuda.max_memory_reserved()/2**30
 st=make_checkpoint(model,opt,scheduler,scaler,batch,schedule);torch.save(st,ckpt/'last.pt');del st
 model.eval();probs=[];vt=[];it=iter(loader(val,vatf,bs));sample_x=None
 with torch.no_grad():
  while True:
   sync();start=time.perf_counter()
   try:x,y,idx=next(it)
   except StopIteration:break
   x=x.cuda(non_blocking=True)
   if sample_x is None:sample_x=x[:2].clone()
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=meta['amp']):logit=binding.scores(model(x),meta['kind'])
   assert torch.isfinite(logit).all();probs.extend(logit.sigmoid().cpu().tolist());sync();vt.append((len(y),time.perf_counter()-start))
  before=model(sample_x);before=binding.scores(before,meta['kind']).clone()
 st=torch.load(ckpt/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(st['model'],strict=True);del st
 model.eval()
 with torch.no_grad():after=binding.scores(model(sample_x),meta['kind'])
 torch.testing.assert_close(before,after,rtol=0,atol=0);checks['checkpoint_prediction_exact']=True
 cost=dict(measured_batches=len(times),warmup_batches=CFG['warmup_batches'],images=sum(n for n,t in times),seconds=sum(t for n,t in times),images_per_second=sum(n for n,t in times)/sum(t for n,t in times),seconds_per_update=sum(t for n,t in times)/len(times),peak_allocated_GiB=peak,peak_reserved_GiB=reserved,epoch_wall_seconds_including_checkpoint=trainsec)
 vcost=dict(batches=len(vt),images=len(val),seconds=sum(t for n,t in vt),images_per_second=len(val)/sum(t for n,t in vt))
 write(out/'training_curve.json',dict(losses=losses,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),unique_originals_presented=len(set(allpresented))))
 return dict(kind=meta['kind'],train_cost=cost,validation_cost=vcost,metrics=metrics_and_scores(out,val,np.asarray(probs)),checks=checks,checkpoint=str(ckpt/'last.pt'),checkpoint_bytes=(ckpt/'last.pt').stat().st_size,checkpoint_sha256=sha(ckpt/'last.pt'),updates=batch,effective_batch=bs,originals_presented=len(allpresented),unique_originals_presented=len(set(allpresented)),precision='BF16 autocast' if meta['amp'] else 'FP32',optimizer=meta['optimizer'],lr=meta['lr'])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--method',required=True,choices=CFG['methods']);a=parser.parse_args();method=a.method
 out=Path(CFG['output_root'])/method;ckpt=Path(CFG['checkpoint_root'])/method;out.mkdir(parents=True,exist_ok=True);ckpt.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():raise RuntimeError('Completed run exists; do not overwrite')
 torch.set_num_threads(4);torch.manual_seed(CFG['seed']);np.random.seed(CFG['seed']);random.seed(CFG['seed']);torch.cuda.manual_seed_all(CFG['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True)
 started=time.perf_counter();write(out/'status.json',dict(status='running',method=method,pid=os.getpid()))
 try:
  train,val=read_data();t=time.perf_counter();model,meta=binding.build(method);model=model.cuda();sync();init_seconds=time.perf_counter()-t
  total=sum(p.numel() for p in model.parameters());tunable=sum(p.numel() for p in model.parameters() if p.requires_grad)
  print('Initialized',method,'seconds',round(init_seconds,2),'parameters',total,'trainable',tunable,flush=True)
  result=frozen_run(method,model,meta,train,val,out,ckpt) if meta['kind']=='features' else train_run(method,model,meta,train,val,out,ckpt)
  result.update(status='passed',method=method,seed=CFG['seed'],epochs=1 if meta['kind']!='features' else None,train_originals=len(train),validation_originals=len(val),total_parameters=total,trainable_parameters=tunable,initialization_seconds=init_seconds,total_wall_seconds=time.perf_counter()-started,protocol_sha256=sha(P/'protocol.json'),implementation_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),P/'models.py']},test_access=False,interpretation='S0 functionality and cost only; not a formal performance comparison')
  write(out/'result.json',result);write(out/'status.json',dict(status='passed',method=method));print('PASSED',method,result['train_cost'],flush=True)
 except BaseException as e:
  write(out/'status.json',dict(status='failed',method=method,error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
