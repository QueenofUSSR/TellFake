"""Freeze the dev-selected protocol and a standalone source tree, then launch."""
import json,os,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from genimage_detector.data import file_hash,load_manifests,load_supplement,write_json

PROJECT=Path(__file__).resolve().parents[1]
QUEUE=Path('/mnt/ssd/data/logs/arin5201/formal_20261113')

def main():
 decision=json.loads(Path('/mnt/ssd/data/logs/arin5201/development_20261113/decision.json').read_text())
 assert decision['chosen_candidate']=='C_js01' and decision['continue_consistency']
 base,fp=load_manifests('data/manifests/v3')
 _,_,supp_fp=load_supplement('data/manifests/v3_supplement_20261113',base,fp)
 if QUEUE.exists():raise FileExistsError(QUEUE)
 # Existing full-budget A seed42 was verified previously; preserve provenance.
 import torch
 a=torch.load('/mnt/ssd/data/ckpt/arin5201/A_v3_s42/best.pt',map_location='cpu',weights_only=True)
 assert not a['smoke_test'] and a['manifest_sha256']==fp and a['config']['epochs']==10 and a['config']['seed']==42
 for tensor in a['state_dict'].values():assert torch.isfinite(tensor).all()
 QUEUE.mkdir();source=QUEUE/'source';source.mkdir()
 for name in ['genimage_detector','configs','scripts','docs','tests','data/manifests/v3','data/manifests/v3_supplement_20261113']:
  shutil.copytree(PROJECT/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for name in ['Dockerfile','requirements-runtime.txt','requirements-local.lock','README.md','IMPLEMENTATION_PLAN.md']:
  shutil.copy2(PROJECT/name,source/name)
 runs=[]
 for seed in [42,43,44]:
  for method,config in [('A','resnet50_A_v3.json'),('B','resnet50_B_v3.json'),('C','resnet50_C_js01_v3.json')]:
   if method=='A' and seed==42:continue
   name=f'{method}_v3_s{seed}_formal_20261113'
   assert not (Path('/mnt/ssd/data/ckpt/arin5201')/name).exists()
   assert not (Path('/mnt/ssd/data/logs/arin5201')/name).exists()
   runs.append(dict(name=name,method=method,config=config,seed=seed))
 hashes={str(p.relative_to(source)):file_hash(p) for p in sorted(source.rglob('*')) if p.is_file()}
 write_json(QUEUE/'source_sha256.json',hashes)
 protocol=dict(stage='formal_frozen_before_final_predictions',seeds=[42,43,44],epochs=10,updates_per_run=2500,train=16000,val=7000,lambda_C=.1,consistency='symmetric Bernoulli Jensen-Shannon',primary_comparison='B vs C',primary_target_fpr=.01,threshold='Validation-only conservative order statistic; fixed across every final source/condition',checkpoint='Max validation AUROC; tie lower BCE',base_manifest_sha256=fp,supplement_manifest_sha256=supp_fp,reused_A_seed42=dict(name='A_v3_s42',checkpoint_sha256=file_hash('/mnt/ssd/data/ckpt/arin5201/A_v3_s42/best.pt'),note='Same data, environment, optimizer, epochs and A transform. Earlier source has less gradient instrumentation; source snapshot retained.'),pending_runs=runs,conditions=['clean','jpeg:50','resize:0.25','blur:1','resize:0.5>jpeg:50','jpeg:50>resize:0.5','jpeg:70>jpeg:50','resize:0.25>jpeg:50>blur:1'],strong_condition='resize:0.25>jpeg:50>blur:1',processing='Pillow RGB EXIF orientation; bicubic shrink/restore; JPEG4:2:0 nonprogressive optimize=False; deterministic parameters. Simulated processing, not platform test.',metrics=['AUROC','balanced_accuracy','AI_recall','actual_real_FPR'],statistics='All three seeds retained; mean/std and per-source/worst-group results. B/C paired on same original images. Any sample bootstrap clusters all condition views of the same original. COCO separately FPR-only.',final_predictions_run=False,selection_source='development_20261113/screening_protocol.json and decision.json',final_selection_prohibited=True,no_automatic_final_evaluation=True,estimated_minutes_remaining=45)
 write_json(QUEUE/'protocol.json',protocol)
 env=os.environ.copy();env.update(TORCH_HOME='/mnt/ssd/cache/torch',XDG_CACHE_HOME='/mnt/ssd/cache',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(source))
 with (QUEUE/'console.log').open('x') as log:
  process=subprocess.Popen([sys.executable,str(source/'scripts/formal_queue.py')],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write_json(QUEUE/'launch.json',dict(pid=process.pid,source=str(source),command=[sys.executable,str(source/'scripts/formal_queue.py')]))
 print('Formal queue started',process.pid,'8 pending runs; A42 reused')

if __name__=='__main__':main()
