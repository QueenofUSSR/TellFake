import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'reproduction/p1';BASE=ROOT/'reproduction/manifests/v1_20261114'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((ROOT/'reproduction/P0_ACCEPTANCE.json').read_text())
for name,h in receipt['evidence'].items():assert sha(ROOT/name)==h,('P0 drift',name)
protocol=json.loads((BASE/'protocol.json').read_text())
with (BASE/'val.csv').open() as f:rows=list(csv.DictReader(f))
val=[r for gen,n in [('Real',512),('SD14',256),('BigGAN',256)] for r in [r for r in rows if r['generator']==gen][:n]]
P.mkdir(exist_ok=True)
with (P/'validation_S0.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(val[0]));w.writeheader();w.writerows(val)
methods=['frozen_B32_resize','frozen_B16_resize','frozen_B16_crop128','frozen_B16_resize_jpeg','frozen_B16_crop128_jpeg','frozen_L14_resize','effort_native','lora_native','effort','effort_degradation','dual_adapter','dual_ssf','frozen_only','adapter_only','ssf_only','iapl']
config=dict(schema=1,seed=42,train_manifest=str(BASE/'train_S0.csv'),train_sha256=protocol['manifest_sha256']['train_S0'],validation_manifest=str(P/'validation_S0.csv'),validation_sha256=sha(P/'validation_S0.csv'),validation_parent_sha256=protocol['manifest_sha256']['val'],validation_count=1024,validation_note='Fixed source/label-stratified S0 subset of validation only. Diagnostic metrics/thresholds, not formal model selection or final results.',methods=methods,epochs=1,batch_size=16,frozen_batch_size=32,warmup_batches=10,min_measured_batches=100,num_workers=4,final_test_access=False,output_root='/mnt/ssd/data/logs/arin5201/reproduction/p1_20261114',checkpoint_root='/mnt/ssd/data/ckpt/arin5201/reproduction/p1_20261114',scope='Three authors core models, native Thomas preprocessing panel, U Effort baseline/degradation pair, dual branch controls and conditional IAPL. Auxiliary-data experiments and other backbones remain separate.')
(P/'protocol.json').write_text(json.dumps(config,indent=2)+'\n');print('P0 verified; P1 protocol frozen:',len(methods),'methods, 4000 train, 1024 fixed validation')
