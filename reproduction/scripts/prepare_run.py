"""Prepare a reviewable S0 config. Does NOT run training or final evaluation."""
import argparse,json,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from reproduction.protocol import read_manifest,assert_disjoint
p=argparse.ArgumentParser();p.add_argument('--method',choices=['dual_adapter','dual_ssf'],required=True);p.add_argument('--seed',type=int,choices=[42,43,44],default=42);p.add_argument('--track',choices=['U','R'],default='U');a=p.parse_args()
R=ROOT/'reproduction';weights=json.loads((R/'weights.lock.json').read_text());clip=next(x['local_path'] for x in weights if x['repo']=='openai/clip-vit-large-patch14')
manifest=R/'manifests'/('v1_20261114' if a.track=='U' else 'R_SD14');train=read_manifest(manifest/'train_S0.csv');val=read_manifest(manifest/'val.csv');assert_disjoint(train,val)
assert {r['generator'] for r in train}==({'Real','SD14','BigGAN'} if a.track=='U' else {'Real','SD14'})
cfg=yaml.safe_load((ROOT/'third_party/clip_dual_aigi/config.yaml').read_text());run=f'{a.track}_{a.method}_S0_s{a.seed}'
cfg.update(protocol='reproduction_'+a.track,frozen_manifest_dir=str(manifest),train_manifest='train_S0',val_manifest='val',evaluate_final=False,clip_path=clip,seed=a.seed,epochs=2,train_generator='SD14' if a.track=='R' else 'SD14+BigGAN',test_generators=['SD14','BigGAN','ADM','GLIDE','SD15','VQDM','Midjourney_subset'],manifest_dir=f'/mnt/ssd/data/cache/arin5201/reproduction/runs/{run}/manifests',checkpoints_dir=f'/mnt/ssd/data/ckpt/arin5201/reproduction/{run}',logs_dir=f'/mnt/ssd/data/logs/arin5201/reproduction/{run}')
path=R/'configs'/(run+'.yaml');path.write_text(yaml.safe_dump(cfg,sort_keys=False));print(path)
