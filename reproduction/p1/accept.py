"""Verify P1 evidence without launching training or touching a final-test model pipeline."""
import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'reproduction/p1';C=json.loads((P/'protocol.json').read_text());E=json.loads((P/'extended_protocol.json').read_text());LOG=Path(C['output_root'])
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 prior=json.loads((ROOT/'reproduction/P0_ACCEPTANCE.json').read_text())
 for name,h in prior['evidence'].items():assert sha(ROOT/name)==h,('P0 drift',name)
 proof=json.loads((P/'verification.json').read_text());assert proof['status']=='passed' and len(proof['methods'])==24 and proof['score_rows_verified']==24576
 delta=json.loads((P/'delta_validation.json').read_text());assert delta['status']=='passed' and len(delta['methods'])==18
 for r in delta['methods']:assert r['status']=='passed' and Path(r['file']).stat().st_size==r['bytes']
 for name,n in [('tests.log',9),('p0_regression_tests.log',10),('legacy_regression_tests.log',25)]:
  s=(LOG/name).read_text();assert f'Ran {n} tests' in s and s.rstrip().endswith('OK')
 assert '"status": "passed"' in (LOG/'data_recheck.log').read_text()
 for method in C['methods']+E['methods']:
  result=json.loads((LOG/method/'result.json').read_text());assert result['status']=='passed' and result['train_cost']['measured_batches']>=100
  for name,h in result['implementation_sha256'].items():
   assert sha(ROOT/name)==h or ((P/'source_versions'/(h+'.py')).is_file() and sha(P/'source_versions'/(h+'.py'))==h),(method,name)
 q=json.loads((P/'s1_s2_queue.json').read_text());assert len(q['recipes'])==150 and len({r['id'] for r in q['recipes']})==150
 for r in q['recipes']:
  assert r['status']=='frozen_not_launched' and not r['final_test_access'];assert sha(r['train_manifest'])==r['train_manifest_sha256']
  for name,h in r['method_code_sha256'].items():assert sha(P/name)==h
 free=shutil.disk_usage('/mnt/ssd').free/2**30;required=q['checkpoint_storage_GiB_with_20pct_retry']+q['feature_and_score_reserve_GiB'];assert required<free
 source_checks=[]
 for lock in ['sources.lock.json','external_baselines.lock.json']:
  for r in json.loads((ROOT/'third_party'/lock).read_text()):
   commit=subprocess.check_output(['git','-C',r['local_path'],'rev-parse','HEAD'],text=True).strip();dirty=subprocess.check_output(['git','-C',r['local_path'],'status','--porcelain'],text=True).strip();assert commit==r['commit'] and not dirty;source_checks.append(r['name'])
 artifacts=[p for p in P.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='P1_ACCEPTANCE.json']
 outputs=[p for method in C['methods']+E['methods']+['iapl_tta'] for p in (LOG/method).glob('*') if p.is_file() and p.suffix in ['.json','.csv']]
 receipt=dict(status='passed',date='2026-11-14',stage='P1/S0 method interfaces and measured costs',method_configurations=24,neural_methods_with_full_and_delta_checkpoints=18,TTA_originals=32,TTA_conditions=2,unit_and_regression_tests=44,validation_predictions_independently_recomputed=24576,formal_training_started=False,final_test_evaluation_started=False,P0_artifacts_unchanged=True,upstream_worktrees_clean=source_checks,available_SSD_GiB=free,planned_additional_storage_GiB=required,S1_S2_frozen_runs=150,S1_S2_gpu_hours_with_retry=q['gpu_hours_with_20pct_retry'],evidence={str(p.relative_to(ROOT)):sha(p) for p in artifacts},external_output_hashes={str(p):sha(p) for p in outputs},limitations=['S0 validation metrics are diagnostic, not formal performance claims','S1/S2 cost extrapolation includes 20% retry; P3/TTA final inference cost separate','Full original R-track data construction/auxiliary sources not complete','Formal multi-epoch queue interruption handling belongs to P2; S0 optimizer/RNG step replay and checkpoint state reconstruction verified'])
 (P/'P1_ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2)+'\n');print('P1 ACCEPTED:',24,'methods;',44,'tests;',round(q['gpu_hours_with_20pct_retry'],1),'GPU hours planned; no P2/final evaluation launched')
if __name__=='__main__':main()
