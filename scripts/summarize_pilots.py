"""Verify pilot predictions, apply the preregistered dev-only candidate rule."""
import csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from genimage_detector.data import load_manifests,read_csv,write_json,write_csv
from genimage_detector.metrics import grouped_metrics,choose_fpr_threshold

ROOT=Path('/mnt/ssd/data/logs/arin5201');OUT=ROOT/'development_20261113'
METHODS=['B','C_js01','C_js1']

def main():
 base,fingerprints=load_manifests('data/manifests/v3');expected={r['path'] for r in base['dev']}
 protocol=json.loads((OUT/'screening_protocol.json').read_text());conditions=protocol['conditions']
 aggregate={};allrows=[];predictions={}
 for method in METHODS:
  name=method+'_v3_pilot_s42_20261113';run=ROOT/name
  history=[json.loads(s) for s in (run/'history.jsonl').read_text().splitlines()]
  assert len(history)==2 and sum(r['updates'] for r in history)==500
  assert json.loads((run/'manifest_sha256.json').read_text())==fingerprints
  val=read_csv(run/'best_val_predictions.csv');threshold=choose_fpr_threshold([int(r['label']) for r in val],[float(r['p_ai']) for r in val],.01)
  assert len(val)==7000
  report=json.loads((ROOT/(name+'_dev')/'metrics.json').read_text());assert report['threshold']==threshold and not report['smoke_test']
  values=[];predictions[method]={}
  for condition in conditions:
   rows=read_csv(ROOT/(name+'_dev')/('dev_'+condition.replace(':','_')+'.csv'))
   assert len(rows)==4000 and {r['path'] for r in rows}==expected
   labels=[int(r['label']) for r in rows];scores=[float(r['p_ai']) for r in rows];generators=[r['generator'] for r in rows]
   metrics=grouped_metrics(labels,scores,generators,threshold)
   assert metrics==report['results']['dev'][condition]
   values.append(metrics['overall']);predictions[method][condition]={r['path']:r for r in rows}
   allrows.append(dict(method=method,condition=condition,**metrics['overall']))
  aggregate[method]=dict(mean_balanced_accuracy=float(np.mean([v['balanced_accuracy'] for v in values])),mean_real_fpr=float(np.mean([v['real_fpr'] for v in values])),mean_recall=float(np.mean([v['recall'] for v in values])),clean_balanced_accuracy=values[0]['balanced_accuracy'],clean_real_fpr=values[0]['real_fpr'],threshold=threshold,epochs=2,updates=500,mean_epoch_seconds=float(np.mean([r['seconds'] for r in history])),peak_GiB=max(r['peak_cuda_allocated_bytes'] for r in history)/1024**3)
 paired=[]
 for method in METHODS[1:]:
  a=aggregate[method];b=aggregate['B'];a['delta_mean_balanced_accuracy_vs_B']=a['mean_balanced_accuracy']-b['mean_balanced_accuracy'];a['delta_mean_fpr_vs_B']=a['mean_real_fpr']-b['mean_real_fpr'];a['passes_guardrails']=a['delta_mean_fpr_vs_B']<=.01 and a['clean_balanced_accuracy']-b['clean_balanced_accuracy']>=-.02
  for condition in conditions:
   corrected=regressed=0
   for path,r in predictions['B'][condition].items():
    other=predictions[method][condition][path]
    assert r['label']==other['label']
    before=int(r['prediction'])==int(r['label']);after=int(other['prediction'])==int(other['label'])
    corrected+=not before and after;regressed+=before and not after
   paired.append(dict(method=method,condition=condition,n_originals=4000,corrected=corrected,regressed=regressed))
 candidates=[m for m in METHODS[1:] if aggregate[m]['passes_guardrails']]
 chosen=max(candidates,key=lambda m:aggregate[m]['mean_balanced_accuracy']) if candidates else None
 signal=chosen is not None and aggregate[chosen]['delta_mean_balanced_accuracy_vs_B']>0
 decision=dict(stage='two_epoch_single_seed_development_screen',final_results_opened=False,aggregate=aggregate,chosen_candidate=chosen,continue_consistency=signal,reason='Candidate maximizes mean dev balanced accuracy subject to prespecified FPR/clean guardrails. Preliminary development signal only; no generalization/statistical-significance claim.',next_step='Freeze chosen lambda and equal 10-epoch three-seed queue before final evaluation' if signal else 'Retain negative/mixed result and analyze dev failure before further methods')
 write_json(OUT/'decision.json',decision);write_csv(OUT/'conditions.csv',allrows,tuple(allrows[0]));write_csv(OUT/'paired_changes.csv',paired,tuple(paired[0]))
 print(json.dumps(decision,indent=2))

if __name__=='__main__':main()
