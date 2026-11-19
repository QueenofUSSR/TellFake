"""Derive R-track lists from already frozen U/reserve identities; never scan sources."""
import csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from reproduction.protocol import read_manifest,assert_disjoint
if (ROOT/'reproduction/P0_ACCEPTANCE.json').exists():raise RuntimeError('Accepted manifests are immutable; create a new version instead.')
BASE=ROOT/'reproduction/manifests/v1_20261114';P=json.loads((BASE/'protocol.json').read_text())
train=read_manifest(BASE/'train_S3.csv');adm=read_manifest(BASE/'R_ADM_train_pool.csv');val=read_manifest(BASE/'val.csv')
pools={g:[r for r in train if r['generator']==g] for g in ['Real','SD14','BigGAN']};pools['ADM']=adm
for name,gens in [('R_SD14',['SD14']),('R_BigGAN',['BigGAN']),('R_BigGAN_ADM',['BigGAN','ADM'])]:
 dest=BASE.parent/name;dest.mkdir(exist_ok=True);splits={}
 for scale,n in [('S0',4000),('S1',16000),('S2',64000)]:
  rows=pools['Real'][:n//2]+[r for g in gens for r in pools[g][:n//(2*len(gens))]]
  assert len(rows)==n;splits['train_'+scale]=[dict(r,split='train_'+scale) for r in rows]
 # R BigGAN+ADM shares the BigGAN-only validation of its single-source control.
 # This protocol deviation is explicit and fixed before any result is inspected.
 validation_fake='SD14' if gens==['SD14'] else 'BigGAN'
 splits['val']=[dict(r,split='val') for g in ['Real',validation_fake] for r in [r for r in val if r['generator']==g][:2500]]
 splits['test_GLIDE']=read_manifest(BASE/'test_GLIDE.csv')+read_manifest(BASE/'test_real.csv')
 assert_disjoint(splits['train_S2'],splits['val'],splits['test_GLIDE'])
 for split,rows in splits.items():
  with (dest/(split+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 protocol=dict(track='R',reproduction_level='method-level protocol-corrected; author image selection not recovered',sources=gens,validation_sources=['Real',validation_fake],selection_limitation='BigGAN-only fake validation for BigGAN+ADM; recipe must be predeclared, no GLIDE tuning' if len(gens)>1 else None,counts={k:len(v) for k,v in splits.items()},source_protocol_sha256=hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest(),manifest_sha256={n:hashlib.sha256((dest/(n+'.csv')).read_bytes()).hexdigest() for n in splits})
 (dest/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print('R method-level S0/S1/S2 manifests prepared')
