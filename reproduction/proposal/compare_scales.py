"""Recompute clean validation comparisons from frozen prediction artifacts."""
import csv,json,math
from pathlib import Path
from reproduction.proposal.run import sha,write
from genimage_detector.metrics import binary_metrics
HERE=Path(__file__).resolve().parent
base=Path('/mnt/ssd/data/logs/arin5201/reproduction');report=[]
for method in ['frozen_L14_resize','effort','effort_degradation']:
 row={'method':method,'scales':{}}
 for scale,folder in [('16k','proposal_round1_20261115' if method.startswith('frozen') else 'proposal_extension10_20261117'),('64k','proposal_scale64k_20261117')]:
  d=base/folder/method;r=json.loads((d/'result.json').read_text());assert r['status']=='complete' and not r['test_access']
  for p,h in r['artifacts'].items():assert sha(p)==h,p
  with (d/'validation_scores.csv').open() as f:rows=list(csv.DictReader(f))
  assert len(rows)==10000 and len({x['original_id'] for x in rows})==10000
  identities=[(x['original_id'],x['source'],x['label']) for x in rows]
  if scale=='16k':reference=identities
  else:assert identities==reference
  for k,e in r['metrics'].items():
   got=binary_metrics([int(x['label']) for x in rows],[float(x['score']) for x in rows],e['threshold'])
   for name,v in e.items():assert v is None and got[name] is None or math.isclose(v,got[name],rel_tol=1e-10,abs_tol=1e-12)
  row['scales'][scale]={'metrics':r['metrics'],'by_source':r['by_source'],'best_epoch':r['detail'].get('best_epoch'),'result':str(d/'result.json'),'result_sha256':sha(d/'result.json')}
 row['delta_percentage_points']={k:100*(row['scales']['64k']['metrics']['p05'][k]-row['scales']['16k']['metrics']['p05'][k]) for k in ['auroc','balanced_accuracy','real_fpr']}
 report.append(row);print(method,'verified',row['delta_percentage_points'],flush=True)
write(HERE/'clean_scale_comparison.json',{'scope':'same 10000 seen validation originals, seed 42; fixed 0.5 comparison; no final test','limitations':'same epoch budget increases both data and updates; best selected on this validation; single seed','comparisons':report,'test_access':False})
