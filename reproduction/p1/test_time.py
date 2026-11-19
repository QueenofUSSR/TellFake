"""Single-GPU native IAPL TTA/OIS. Every sample starts from identical state.

No test manifest is opened. Uses 32 fixed validation originals, clean and JPEG70.
This is a functional/cost probe, not formal TTA thresholds or final evaluation.
"""
import contextlib,copy,csv,hashlib,io,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageOps
from reproduction.p1 import models as binding
from genimage_detector.metrics import choose_threshold,choose_fpr_threshold,binary_metrics
P=ROOT/'reproduction/p1';C=json.loads((P/'protocol.json').read_text());OUT=Path(C['output_root'])/'iapl_tta';OUT.mkdir(exist_ok=True)
def sha_tensor(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
class LocalTTA(torch.nn.Module):
 def __init__(self,base):super().__init__();self.module=base
 def forward(self,x):return self.module(x)
 def no_sync(self):return contextlib.nullcontext()
def main():
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
 args=binding.iapl_args(True)
 from models import build_model
 # Import native module by filename to avoid this P1 module's name shadowing it.
 native=binding.module('iapl_native_test_time',binding.SRC/'IAPL/test_time.py')
 from utils.dataset import Dataset_Creator_GenImage
 model=build_model(args).cuda();state=torch.load(Path(C['checkpoint_root'])/'iapl/last.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True);del state;model.freeze_tta()
 wrapped=LocalTTA(model);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.005)
 optim_initial=copy.deepcopy(optimizer.state_dict())
 adaptable={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad};buffers={n:b.detach().clone() for n,b in model.named_buffers()}
 assert list(adaptable)==['prompt_learner.ctx'],list(adaptable)
 frozen_hashes={n:sha_tensor(p) for n,p in model.named_parameters() if not p.requires_grad}
 tf=Dataset_Creator_GenImage('',32).transforms['tta']
 with open(C['validation_manifest']) as f:rows=list(csv.DictReader(f))
 rows=[r for source,n in [('Real',16),('SD14',8),('BigGAN',8)] for r in [r for r in rows if r['generator']==source][:n]]
 def reset():
  with torch.no_grad():
   for n,p in model.named_parameters():
    if n in adaptable:p.copy_(adaptable[n])
   for n,b in model.named_buffers():b.copy_(buffers[n])
  optimizer.load_state_dict(copy.deepcopy(optim_initial))
  assert not optimizer.state
  assert all(torch.equal(dict(model.named_parameters())[n],v) for n,v in adaptable.items())
  assert all(torch.equal(dict(model.named_buffers())[n],v) for n,v in buffers.items())
 def predict(row,condition,ois=True):
  seed=int.from_bytes(hashlib.sha256((row['original_id']+'|'+condition+'|42').encode()).digest()[:4],'big');random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  torch.cuda.synchronize();start=time.perf_counter();reset()
  with Image.open(row['path']) as im:im.load();im=ImageOps.exif_transpose(im).convert('RGB')
  if condition=='jpeg70':
   b=io.BytesIO();im.save(b,'JPEG',quality=70,subsampling=2);b.seek(0);im=Image.open(b).convert('RGB')
  images=torch.stack(tf(im)).cuda();assert len(images)==32;args.ois=ois;wrapped.train()
  with torch.autocast('cuda',dtype=torch.bfloat16):index=native.test_time_tuning(wrapped,images,optimizer,None,args)
  wrapped.eval()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   if ois:
    pred=model(images[index]).sigmoid();pred=pred[torch.max(torch.abs(pred-.5),dim=0)[1]]
   else:pred=model(images[:1]).sigmoid()
  assert torch.isfinite(pred).all();torch.cuda.synchronize()
  return float(pred.flatten()[0]),time.perf_counter()-start
 torch.cuda.reset_peak_memory_stats();results=[]
 # Independence probe: A -> B -> A must give the same A, including mutable buffers.
 a,_=predict(rows[0],'clean');predict(rows[-1],'jpeg70');again,_=predict(rows[0],'clean');assert a==again,(a,again)
 for condition in ['clean','jpeg70']:
  for i,r in enumerate(rows):
   score,sec=predict(r,condition);results.append(dict(path=r['path'],original_id=r['original_id'],source=r['generator'],label=int(r['label']),condition=condition,score=score,seconds=sec,views=32,adapt_steps=2,backward_calls=2,forward_calls=3,forward_image_presentations=64+int(32*args.selection_p)))
   if i%8==0:print('TTA',condition,i,'latency',round(sec,3),flush=True)
 assert len(results)==64 and len({(r['original_id'],r['condition']) for r in results})==64
 for n,p in model.named_parameters():
  if n in frozen_hashes:assert sha_tensor(p)==frozen_hashes[n],n
 # Exercise OIS-off using the native include-original view rule; not a result panel.
 noois,sec=predict(rows[0],'clean',False);reset()
 with (OUT/'validation_probe_scores.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 timings=[r['seconds'] for r in results]
 clean=[r for r in results if r['condition']=='clean'];y=[r['label'] for r in clean];scores=[r['score'] for r in clean]
 thresholds={'p05':.5,'max_BA':choose_threshold(y,scores),'FPR01':choose_fpr_threshold(y,scores)}
 probe_threshold_metrics={k:binary_metrics(y,scores,v) for k,v in thresholds.items()}
 report=dict(probe_thresholds=thresholds,probe_threshold_metrics=probe_threshold_metrics,status='passed',scope='S0 validation-only TTA/OIS/reset/cost probe; 32 originals x 2 conditions, not final results or full-validation thresholds',samples=64,originals=32,originals_by_source={'Real':16,'SD14':8,'BigGAN':8},views=32,steps=2,selection_p=args.selection_p,forward_calls_per_sample=3,backward_calls_per_sample=2,mean_seconds_per_image=float(np.mean(timings)),median_seconds_per_image=float(np.median(timings)),p95_seconds_per_image=float(np.percentile(timings,95)),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,checks=dict(single_gpu_coverage=True,A_B_A_exact=True,optimizer_reset_each_sample=True,parameters_and_buffers_reset_each_sample=True,frozen_parameters_unchanged=True,ois_off_finite=True),ois_off_probe_score=noois,train_checkpoint=str(Path(C['checkpoint_root'])/'iapl/last.pt'),thresholds='Not frozen on this 32-original cost probe. Formal TTA thresholds require same TTA pipeline on full validation.',test_access=False)
 (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('TTA PASSED',report,flush=True)
if __name__=='__main__':main()
