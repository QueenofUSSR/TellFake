"""Fork completed epoch-5 states, changing only provenance/config identity."""
import copy, hashlib, json, shutil
from pathlib import Path
import numpy as np
import torch
from reproduction.proposal.run import sha, write, atomic_save
HERE=Path(__file__).resolve().parent
parent=json.loads((HERE/'round1.json').read_text())
fingerprint=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
parent_fp=fingerprint(parent)
for p,h in parent['implementation_sha256'].items():assert sha(p)==h,p
cfg=copy.deepcopy(parent)
cfg.update(epochs=10,methods=['effort','effort_degradation'],experiment_id='S1_s42_effort_pair_extension10_20261117',parent_config=str(HERE/'round1.json'),parent_config_sha256=parent_fp,continuation='Resume epoch-5 last checkpoint including optimizer/scheduler/scaler/RNG; select best over epochs 1-10; no final test',preparation_sha256=sha(__file__))
for k in ['output_root','checkpoint_root','cache_root']:
 cfg[k]=cfg[k].replace('proposal_round1_20261115','proposal_extension10_20261117')
new_fp=fingerprint(cfg)
config_path=HERE/'extension10.json'
assert not config_path.exists(),'Extension already registered; do not overwrite'
for k in ['output_root','checkpoint_root']:assert not Path(cfg[k]).exists(),'Extension output exists'

def same(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:same(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):same(x,y)
 else:assert a==b

for method in cfg['methods']:
 src=Path(parent['output_root'])/method; dst=Path(cfg['output_root'])/method
 cs=Path(parent['checkpoint_root'])/method;cd=Path(cfg['checkpoint_root'])/method
 result=json.loads((src/'result.json').read_text())
 assert result['status']=='complete' and result['config_sha256']==parent_fp and not result['test_access']
 for p,h in result['artifacts'].items():assert sha(p)==h,p
 dst.mkdir(parents=True);cd.mkdir(parents=True)
 receipt={'parent_result':str(src/'result.json'),'parent_result_sha256':sha(src/'result.json'),'parent_config_sha256':parent_fp,'config_sha256':new_fp,'checkpoints':{}}
 for name in ['last.pt','best.pt']:
  source_hash=sha(cs/name)
  st=torch.load(cs/name,map_location='cpu',weights_only=False)
  assert st['config_sha256']==parent_fp
  if name=='last.pt':
   assert (st['epoch'],st['offset'],st['updates'])==(5,0,5000)
   assert len(st['curve'])==5 and st['scheduler']['last_epoch']==5
   assert all(g['lr']==.0002 for g in st['optimizer']['param_groups'])
  st['config_sha256']=new_fp
  atomic_save(st,cd/name)
  loaded=torch.load(cd/name,map_location='cpu',weights_only=False);same(st,loaded)
  assert sha(cs/name)==source_hash
  receipt['checkpoints'][name]={'parent_sha256':source_hash,'initial_child_sha256':sha(cd/name),'state_roundtrip_exact':True,'epoch':st['epoch'],'updates':st['updates']}
  del st,loaded
 for name in ['curve.json','best_validation_probabilities.npy']+[f'validation_epoch_{i}.csv' for i in range(1,6)]:
  shutil.copy2(src/name,dst/name);assert sha(src/name)==sha(dst/name)
 write(dst/'continuation_provenance.json',receipt)
 print(method,'prepared verified epoch-5 continuation',flush=True)
write(config_path,cfg)
print('Registered',config_path,flush=True)
