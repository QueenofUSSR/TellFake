"""Recover numerical overflow, verify nine runs, then frozen final evaluation."""
import json,os,shutil,subprocess,sys
from pathlib import Path
import torch
from genimage_detector.data import load_manifests,load_supplement,file_hash,read_csv,write_json
from genimage_detector.metrics import binary_metrics,choose_fpr_threshold
ROOT=Path('/mnt/ssd/data/logs/arin5201');OUT=ROOT/'final_20261113';PROJECT=Path('/mnt/ssd/code/arin5201')

def main():
 OUT.mkdir(exist_ok=False)
 source=OUT/'recovery_source';shutil.copytree(ROOT/'formal_20261113/source',source,ignore=shutil.ignore_patterns('__pycache__'))
 shutil.copy2(PROJECT/'genimage_detector/engine.py',source/'genimage_detector/engine.py')
 shutil.copy2(PROJECT/'tests/test_amp_replay.py',source/'tests/test_amp_replay.py')
 write_json(OUT/'recovery_protocol.json',dict(reason='B seed44 failed on nonfinite unscaled gradients after epoch6; finite forward loss. AMP overflow handled by reducing scale and replaying same batch with BN/RNG rollback; persistent nonfinite gradients still fail. No optimizer update dropped.',failed_run='B_v3_s44_formal_20261113',retained_failed_artifacts=True,restart='B44 from initialization, same seed/data/10 epochs; C44 uses same guarded implementation',hyperparameters_changed=False,tests='25 tests passed, including injected-overflow replay vs finite reference',final_predictions_not_yet_opened=True))
 protocol=json.loads((ROOT/'formal_20261113/protocol.json').read_text())
 runs=[dict(name='A_v3_s42',method='A',seed=42)]+[dict(r) for r in protocol['pending_runs']]
 env=os.environ.copy();env.update(TORCH_HOME='/mnt/ssd/cache/torch',XDG_CACHE_HOME='/mnt/ssd/cache',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(source))
 for run in runs:
  if run['seed']!=44 or run['method']=='A':continue
  run['name']=run['method']+'_v3_s44_recovery_20261113'
  write_json(OUT/'status.json',dict(stage='recovery_training',current=run['name']))
  cmd=[sys.executable,'-u','-m','genimage_detector','train','--config',str(source/'configs'/run['config']),'--manifest-dir',str(source/'data/manifests/v3'),'--data-root','/mnt/ssd/data/datasets/GenImage/official_v3','--name',run['name'],'--seed','44','--device','cuda']
  with (OUT/(run['name']+'.log')).open('x') as log:subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 base,fp=load_manifests(PROJECT/'data/manifests/v3');_,_,sfp=load_supplement(PROJECT/'data/manifests/v3_supplement_20261113',base,fp)
 assert fp==protocol['base_manifest_sha256'] and sfp==protocol['supplement_manifest_sha256']
 for run in runs:
  log=ROOT/run['name'];history=[json.loads(l) for l in (log/'history.jsonl').read_text().splitlines()];assert len(history)==10 and sum(r['updates'] for r in history)==2500
  ckpt=Path('/mnt/ssd/data/ckpt/arin5201')/run['name']/'best.pt';c=torch.load(ckpt,map_location='cpu',weights_only=True)
  assert not c['smoke_test'] and c['manifest_sha256']==fp
  predictions=read_csv(log/'best_val_predictions.csv');assert len(predictions)==7000 and {r['path'] for r in predictions}=={r['path'] for r in base['val']}
  y=[int(r['label']) for r in predictions];p=[float(r['p_ai']) for r in predictions]
  assert choose_fpr_threshold(y,p,.01)==c['threshold'] and binary_metrics(y,p,c['threshold'])==c['val_metrics']
  for n,h in c['source_sha256'].items():assert file_hash(log/'source'/n)==h
  run.update(checkpoint=str(ckpt),checkpoint_sha256=file_hash(ckpt),threshold=c['threshold'],best_epoch=c['epoch'],amp_replays=sum(h.get('amp_batch_retries',0) for h in history),evaluation=run['name']+'_final')
 write_json(OUT/'verified_runs.json',runs)
 write_json(OUT/'evaluation_protocol.json',dict(conditions=protocol['conditions'],base_fingerprints=fp,supplement_fingerprints=sfp,thresholds='Frozen per checkpoint, never refit on tests',recovery='recovery_protocol.json',bootstrap_seed=20261113,bootstrap_replicates=1000,bootstrap_unit='Original image; condition views and matched B/C predictions stay together. Seeds averaged first; seed variation separately reported. Pointwise 95% percentile intervals, not multiplicity corrected.'))
 for i,run in enumerate(runs):
  write_json(OUT/'status.json',dict(stage='final_evaluation',current=run['name'],completed_evaluations=i,total=9))
  cmd=[sys.executable,'-u','-m','genimage_detector','evaluate','--manifest-dir',str(source/'data/manifests/v3'),'--supplement-dir',str(source/'data/manifests/v3_supplement_20261113'),'--data-root','/mnt/ssd/data/datasets/GenImage/official_v3','--checkpoint',run['checkpoint'],'--splits','seen_test','unseen_test','midjourney_test','external_real_test','--name',run['evaluation'],'--conditions',*protocol['conditions'],'--device','cuda']
  with (OUT/(run['evaluation']+'.log')).open('x') as log:subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 write_json(OUT/'status.json',dict(stage='evaluations_completed',completed_evaluations=9,total=9))

if __name__=='__main__':
 try:main()
 except BaseException as e:
  write_json(OUT/'failure.json',dict(error=repr(e)));raise
