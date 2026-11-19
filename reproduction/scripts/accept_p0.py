"""Fail-closed acceptance for core P0; explicitly preserve blocked optional scopes."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reproduction';LOG=Path('/mnt/ssd/data/logs/arin5201/reproduction/p0_20261114')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 proofs=['data_validation.json','environment_validation.json','iapl_environment_validation.json','source_validation.json']
 for name in proofs:
  d=json.loads((R/name).read_text());assert d['status']=='passed',name
 data=json.loads((R/'data_validation.json').read_text());assert data['protocol_sha256']==sha(R/'manifests/v1_20261114/protocol.json')
 for protocol in (R/'manifests').glob('*/protocol.json'):
  p=json.loads(protocol.read_text())
  for name,h in p['manifest_sha256'].items():assert sha(protocol.parent/(name+'.csv'))==h
 weights=json.loads((R/'weights.lock.json').read_text());assert len(weights)==5
 for model in weights:
  for f in model['files']:
   assert sha(Path(f['path']))==f['sha256']
   if f.get('upstream_lfs_sha256'):assert f['sha256']==f['upstream_lfs_sha256']
 for env in json.loads((R/'environments.json').read_text()):
  assert sha(ROOT/env['lock'])==env['lock_sha256']
  subprocess.run(['uv','pip','check','--python',env['path']+'/bin/python'],check=True,capture_output=True)
 for name,count in [('tests.log',10),('legacy_tests.log',25)]:
  s=(LOG/name).read_text();assert f'Ran {count} tests' in s and s.rstrip().endswith('OK'),name
 patches=json.loads((R/'patches/registry.json').read_text())
 for p in patches:
  if p.get('patch'):assert sha(ROOT/p['patch'])==p['patch_sha256']
 matrix=json.loads((R/'experiment_matrix.json').read_text());assert len(matrix['rows'])==209
 for row in matrix['rows']:
  if row.get('data_sha256'):
   assert row['data_sha256']==sha(R/'manifests/v1_20261114'/(row['train_manifest']+'.csv'))
 assert json.loads((R/'iapl_environment_validation.json').read_text())['conditional_ctx'] is True
 files=[p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='P0_ACCEPTANCE.json']
 receipt=dict(status='passed',date='2026-11-14',scope='Core P0 data/environment/protocol preparation; not all R-track auxiliary data and not P1/S0 completion',training_started=False,final_model_scores_read=False,evidence={str(p.relative_to(ROOT)):sha(p) for p in files},tests={'reproduction':10,'legacy':25},data={'retained_inventory':894000,'candidate_full_decode':302976,'legacy_v2_additional_decode':14000,'historical_excluded':56995,'unique_frozen_including_R_reserves':197000,'U_train_scales':[4000,16000,64000,128000],'val':10000,'dev':8000,'confirmation_unique_images':19000},remaining_dependencies=['RAISE/Mirage/EvalGEN/Wukong for corresponding full R experiments','ADM external archive SHA256 receipt unavailable; local traceability and hashes retained','Full method S0/AMP/parameter/restore/score/TTA tests and at least 100 measured batches in P1','Formal S1/S2 queue recipes and budget freeze after P1','Optional GAPL external data and additional backbone weights'],limitations=['Candidate-scope pixel audit, not the entire reserve','Conservative dHash false positives; no complete crop/semantic dedup guarantee','Same-domain confirmation despite excluding previously used originals'])
 (R/'P0_ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2)+'\n');print('CORE P0 ACCEPTED; 35 tests; no training started')
if __name__=='__main__':main()
