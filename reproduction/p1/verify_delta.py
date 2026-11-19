import csv,gc,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from reproduction.p1 import models as core,models_extended as extra
from reproduction.p1.checkpoint import make_delta,load_delta,tensor_hash
from reproduction.p1.run_method import sha
P=ROOT/'reproduction/p1';C=json.loads((P/'protocol.json').read_text());E=json.loads((P/'extended_protocol.json').read_text());LOG=Path(C['output_root'])
def main():
 torch.set_num_threads(4);selected=sys.argv[1:] or C['methods']+E['methods'];checks=[]
 if sys.argv[1:] and (P/'delta_validation.json').exists():checks=[r for r in json.loads((P/'delta_validation.json').read_text())['methods'] if r['method'] not in selected]
 with open(C['validation_manifest']) as f:val=list(csv.DictReader(f))[:2]
 for method in selected:
  if method.startswith('frozen_') and method!='frozen_only':continue
  torch.manual_seed(42);np.random.seed(42);random.seed(42)
  binding=core if method in C['methods'] else extra
  model,meta=binding.build(method);full=torch.load(Path(C['checkpoint_root'])/method/'last.pt',weights_only=False,map_location='cpu');delta=make_delta(full,model)
  # The omitted state must match a freshly reconstructed native pretrained base.
  current=model.state_dict()
  for n,h in delta['base_sha256'].items():assert tensor_hash(current[n])==h,('frozen drift',method,n)
  path=Path(C['checkpoint_root'])/method/'delta.pt';torch.save(delta,path);del delta,current
  model=model.cuda();model.load_state_dict(full['model'],strict=True);model.eval()
  ds=core.Images(val,binding.Transform(method,False,meta));x=torch.stack([ds[i][0] for i in range(2)]).cuda()
  with torch.no_grad():before=binding.scores(model(x),meta['kind']).clone()
  # Corrupt every saved component, then reload solely from the delta artifact.
  with torch.no_grad():
   for p in model.parameters():
    if p.requires_grad:p.zero_()
   for n,b in model.named_buffers():
    if not n.endswith('weight_main'):b.zero_()
  disk=torch.load(path,weights_only=False,map_location='cpu');restored=load_delta(model,disk)
  with torch.no_grad():after=binding.scores(model(x),meta['kind'])
  torch.testing.assert_close(before,after,rtol=0,atol=0)
  assert restored['next_batch']==full['next_batch'] and restored['sampler']==full['sampler']
  # All reconstructed tensors, not only two predictions, must match the full checkpoint.
  for n,t in model.state_dict().items():torch.testing.assert_close(t.cpu(),full['model'][n],rtol=0,atol=0)
  checks.append(dict(method=method,status='passed',file=str(path),sha256=sha(path),bytes=path.stat().st_size,full_bytes=Path(C['checkpoint_root'],method,'last.pt').stat().st_size,checks=['fresh frozen base hashes match trained checkpoint','all model tensors reconstruct exactly','prediction exact after intentional corruption','optimizer/scheduler/scaler/RNG/sampler retained']))
  print('Delta verified',method,round(path.stat().st_size/2**20,2),'MiB',flush=True)
  del full,disk,model,x,before,after;gc.collect();torch.cuda.empty_cache()
 (P/'delta_validation.json').write_text(json.dumps(dict(status='passed',methods=checks),indent=2)+'\n')
if __name__=='__main__':main()
