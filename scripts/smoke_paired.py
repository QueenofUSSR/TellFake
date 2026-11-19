"""GPU-serial integration checks after A, without reading final test images."""
import json
import subprocess
from pathlib import Path
import numpy as np

root=Path('/mnt/ssd/data/logs/arin5201')
state=json.loads((root/'launches/A_v3_s42/status.json').read_text())
if state['status']!='completed':raise RuntimeError('A must finish before GPU smoke checks')
configs=[('B','configs/resnet50_B_v3.json',None),('C0','configs/resnet50_C_js01_v3.json',0.),('C01','configs/resnet50_C_js01_v3.json',None)]
records={}
for key,config,weight in configs:
 name='smoke_'+key+'_v3_s42'
 cmd=['./scripts/run_local.sh','train','--config',config,'--manifest-dir','data/manifests/v3','--name',name,'--epochs','2','--max-train-samples','512','--max-val-samples','256']
 if weight is not None:cmd += ['--consistency-weight',str(weight)]
 subprocess.run(cmd,check=True)
 run=root/name
 records[key]=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
 import csv
 predictions=list(csv.DictReader((run/'best_val_predictions.csv').open()))
 scores=np.array([float(r['p_ai']) for r in predictions])
 if not np.isfinite(scores).all() or scores.std()<1e-5:raise ValueError('Nonfinite/constant predictions')
# lambda=0 must use exactly the same images, stochastic draws, and optimization.
for b,c in zip(records['B'],records['C0']):
 for key in ['train_loss','val_loss','classification_loss','consistency_loss','first_gradient_norm','validation']:
  if b[key]!=c[key]:raise AssertionError(f'B/C0 mismatch: {key}: {b[key]} != {c[key]}')
if (root/'smoke_B_v3_s42/best_val_predictions.csv').read_bytes() != (root/'smoke_C0_v3_s42/best_val_predictions.csv').read_bytes():
 raise AssertionError('B/C0 prediction CSV differs')
(root/'audit_v3_20261112/paired_smoke_summary.json').write_text(json.dumps({'smoke_only':True,'B_equals_C0_values_gradients_predictions':True,'runs':records},indent=2))
print('Paired integration passed: B/C0 identical; C01 finite, nonconstant predictions',flush=True)
