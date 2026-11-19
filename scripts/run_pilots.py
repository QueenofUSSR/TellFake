"""Prespecified equal-budget B/C screening. Only validation and dev predictions."""
import json,subprocess
from pathlib import Path
from genimage_detector.data import write_json,file_hash

ROOT=Path('/mnt/ssd/data/logs/arin5201/development_20261113')
RUNS=[('B','configs/resnet50_B_v3.json'),('C_js01','configs/resnet50_C_js01_v3.json'),('C_js1','configs/resnet50_C_js1_v3.json')]
CONDITIONS=['clean','jpeg:70','resize:0.5','resize:0.5>jpeg:70','jpeg:70>resize:0.5']

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 protocol=ROOT/'screening_protocol.json'
 if protocol.exists():raise FileExistsError(protocol)
 write_json(protocol,dict(stage='pilot_not_final',seed=42,epochs=2,updates_per_run=500,train_images=16000,val_images=7000,dev_images=4000,conditions=CONDITIONS,configs={m:file_hash(c) for m,c in RUNS},selection='Rank C by mean dev balanced accuracy across five conditions; guard mean dev FPR increase <=0.01 absolute and clean balanced accuracy drop <=0.02 against B. Single seed/two epochs is preliminary; no final tests.',final_test_predictions=False))
 for method,config in RUNS:
  name=method+'_v3_pilot_s42_20261113'
  for stage,cmd in [('train',['./scripts/run_local.sh','train','--config',config,'--manifest-dir','data/manifests/v3','--name',name,'--epochs','2']),('dev',['./scripts/run_local.sh','evaluate','--manifest-dir','data/manifests/v3','--checkpoint','/mnt/ssd/data/ckpt/arin5201/'+name+'/best.pt','--splits','dev','--name',name+'_dev','--conditions',*CONDITIONS])]:
   write_json(ROOT/'status.json',dict(method=method,stage=stage,status='running'))
   with (ROOT/(method+'_'+stage+'.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 write_json(ROOT/'status.json',dict(status='completed',final_test_predictions=False))

if __name__=='__main__':
 try:main()
 except BaseException as e:
  write_json(ROOT/'failure.json',dict(error=repr(e)));raise
