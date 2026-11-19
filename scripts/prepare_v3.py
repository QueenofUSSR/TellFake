"""Build a reproducible audited subset from the verified official-layout inventory.

Only prospective experiment images are decoded/pixel-audited; this is not a
pixel/near-duplicate audit of the full reserve. No source image is modified.
"""
import csv
import hashlib
import io
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genimage_detector.data import file_hash, read_csv, write_csv, write_json, validate_v3

ROOT = Path('/mnt/ssd/data/datasets/GenImage/official_v3')
OUTPUT = Path('data/manifests/v3')
AUDIT = Path('/mnt/ssd/data/logs/arin5201/audit_v3_20261112')
FIELDS = ('path','label','generator','split','source_split','width','height','format',
          'sha256','pixel_sha256','source','original_id','group_id','dhash')
SEED = 42


def inspect(row):
    p = ROOT / row['path']
    content = p.read_bytes()
    if hashlib.sha256(content).hexdigest() != row['sha256']:
        raise ValueError('Inventory file changed: '+str(p))
    with Image.open(io.BytesIO(content)) as im:
        im.load()
        fmt = im.format
        rgb = ImageOps.exif_transpose(im).convert('RGB')
        pixels = hashlib.sha256(f'{rgb.width}x{rgb.height}:'.encode()+rgb.tobytes()).hexdigest()
        small = np.array(rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS))
        dhash = int.from_bytes(np.packbits(small[:,1:] > small[:,:-1]).tobytes(),'big')
    return dict(row,width=rgb.width,height=rgb.height,format=fmt,pixel_sha256=pixels,dhash=f'{dhash:016x}')


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    AUDIT.mkdir(parents=True,exist_ok=True)
    rng = random.Random(SEED)
    # Exclude all historical v2 images, including holdouts, conservatively.
    old_file, old_pixel = set(), set()
    for p in Path('data/manifests/v2').glob('*.csv'):
        for row in read_csv(p):
            if row.get('sha256'): old_file.add(row['sha256'])
            if row.get('pixel_sha256'): old_pixel.add(row['pixel_sha256'])
    pools=defaultdict(list)
    db=sqlite3.connect('file:/mnt/ssd/data/logs/arin5201/downloads_v3/inventory_hashes.sqlite?mode=ro',uri=True)
    # Global original-name groups: source ImageNet identity is preserved even
    # if a mirrored copy was reencoded. Official val identities stay holdouts.
    identities={}
    for path,sha,label in db.execute('SELECT path,sha256,label FROM images ORDER BY path'):
        source,archive,split,folder,name=Path(path).parts
        if source not in ('BigGAN','SD15','VQDM'): continue
        if split not in ('train','val') or folder not in ('ai','nature'):
            raise ValueError('Unexpected layout '+path)
        real=folder=='nature'
        if (label=='real') != real: raise ValueError('Label inconsistency')
        if not real and source=='VQDM' and split!='val': continue
        original=('ImageNet/'+Path(name).stem) if real else source+'/'+Path(name).stem
        row=dict(path=path,sha256=sha,label=int(not real),generator='Real' if real else source,
                 source=source,source_split=split,split='',original_id=original)
        previous=identities.get(original)
        if previous is None or (split=='val' and previous['source_split']!='val'):
            identities[original]=row
    seen_sha=set()
    for row in sorted(identities.values(),key=lambda r:(r['source_split']!='val',r['path'])):
        if row['sha256'] in old_file or row['sha256'] in seen_sha: continue
        seen_sha.add(row['sha256'])
        pools[(row['generator'],row['source_split'])].append(row)
    del identities
    # Oversample candidates to allow exact/near grouping and historic exclusions.
    needed={('BigGAN','train'):6000,('SD15','train'):6000,('Real','train'):15000,
            ('BigGAN','val'):1000,('SD15','val'):1000,('VQDM','val'):2000,('Real','val'):5000}
    candidates=[]
    for key,n in needed.items():
        pool=pools[key];rng.shuffle(pool)
        take=min(len(pool),int(n*1.35)+200)
        if take<n: raise ValueError(f'Insufficient {key}: {take}/{n}')
        candidates.extend(pool[:take])
    del pools
    # Resume expensive decoding only when file hash still matches the cache key.
    cache=sqlite3.connect(AUDIT/'candidate_cache.sqlite')
    cache.execute('CREATE TABLE IF NOT EXISTS decoded(path TEXT PRIMARY KEY, sha TEXT, row_json TEXT)')
    existing={p:(sha,json.loads(j)) for p,sha,j in cache.execute('SELECT * FROM decoded')}
    rows=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        for start in range(0,len(candidates),256):
            chunk=candidates[start:start+256]
            pending=[r for r in chunk if r['path'] not in existing or existing[r['path']][0]!=r['sha256']]
            for row in ex.map(inspect,pending):
                existing[row['path']]=(row['sha256'],row)
                cache.execute('INSERT OR REPLACE INTO decoded VALUES(?,?,?)',(row['path'],row['sha256'],json.dumps(row)))
            rows.extend(existing[r['path']][1] for r in chunk)
            cache.commit()
            if start%2560==0: print(f'Decoded {min(start+256,len(candidates))}/{len(candidates)}',flush=True)
    rows=[r for r in rows if r['pixel_sha256'] not in old_pixel]
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[max(a,b)]=min(a,b)
    exact={};bands=[defaultdict(list) for _ in range(5)];pairs=[]
    hashes=[int(r['dhash'],16) for r in rows]
    for i,row in enumerate(rows):
        key=row['pixel_sha256']
        if key in exact:
            j=exact[key]
            if row['label']!=rows[j]['label']: raise ValueError('Pixel label conflict')
            union(i,j)
        else:exact[key]=i
        # Five disjoint bands guarantee retrieval for Hamming distance <=4.
        possible=set()
        for b in range(5):possible.update(bands[b][(hashes[i]>>(13*b))&8191])
        for j in sorted(possible):
            distance=(hashes[i]^hashes[j]).bit_count()
            if distance<=4:
                union(i,j)
                pairs.append(dict(path_a=rows[j]['path'],path_b=row['path'],distance=distance,
                                  same_label=rows[j]['label']==row['label']))
        for b in range(5):bands[b][(hashes[i]>>(13*b))&8191].append(i)
        if i%10000==0:print('Near-duplicate retrieval',i,len(pairs),flush=True)
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[find(i)].append(r)
    write_csv(AUDIT/'near_duplicate_candidates.csv',pairs,('path_a','path_b','distance','same_label'))
    eligible=defaultdict(list);excluded=[]
    # Conservative policy: every connected near-duplicate component contributes
    # at most ONE image; mixed-label or multi-generator AI components excluded.
    for component in groups.values():
        if len({r['label'] for r in component})>1 or len({r['generator'] for r in component if r['label']})>1:
            excluded.extend(r['path'] for r in component);continue
        component.sort(key=lambda r:(r['source_split']!='val',r['path']))
        row=component[0]
        row['group_id']=hashlib.sha256('\n'.join(sorted(r['original_id'] for r in component)).encode()).hexdigest()
        eligible[(row['generator'],row['source_split'])].append(row)
    for pool in eligible.values():rng.shuffle(pool)
    allocations={'train':{('Real','train'):8000,('BigGAN','train'):4000,('SD15','train'):4000},
                 'val':{('Real','train'):5000,('BigGAN','train'):1000,('SD15','train'):1000},
                 'dev':{('Real','train'):2000,('BigGAN','train'):1000,('SD15','train'):1000},
                 'seen_test':{('BigGAN','val'):1000,('SD15','val'):1000},
                 'unseen_test':{('VQDM','val'):2000},'real_test':{('Real','val'):5000}}
    splits={}
    for name,targets in allocations.items():
        result=[]
        for key,n in targets.items():
            if len(eligible[key])<n:raise ValueError(f'Insufficient audited groups {key}: {len(eligible[key])}/{n}')
            result.extend(dict(r,split=name) for r in eligible[key][:n]);del eligible[key][:n]
        rng.shuffle(result);splits[name]=result
    validate_v3(splits)
    OUTPUT.mkdir(parents=True)
    for name,result in splits.items():write_csv(OUTPUT/f'{name}.csv',result,FIELDS)
    distributions={name:dict(Counter(f"{r['generator']}/{r['source_split']}/{r['format']}/{r['width']}x{r['height']}" for r in rs)) for name,rs in splits.items()}
    write_json(AUDIT/'encoding_distributions.json',distributions)
    metadata=dict(schema_version=3,seed=SEED,data_root=str(ROOT),seen_generators=['BigGAN','SD15'],
                  unseen_generators=['VQDM'],reserved_unavailable_generators=['Midjourney'],
                  counts={n:dict(Counter(r['generator'] for r in rs)) for n,rs in splits.items()},
                  manifest_sha256={n:file_hash(OUTPUT/f'{n}.csv') for n in splits},
                  audit=dict(candidate_count=len(candidates),decoded_count=len(rows),near_pairs=len(pairs),
                             excluded_mixed_component_images=len(excluded),historical_v2_excluded=True,
                             near_duplicate_method='64-bit dHash Hamming <=4; entire connected components contribute at most one image',
                             limitations=['Candidate-subset pixel/near audit, not full reserve audit',
                                          'dHash cannot guarantee absence of crops or all derived images',
                                          'Source layout/metadata provide labels; no independent content provenance certification',
                                          'One unseen generator available; Midjourney and 5000 external real images still missing',
                                          'Random image sampling; no verified semantic class mapping across generators']),
                  policy='Official train only for train/val/dev; official val only for tests. One global original-identity pool; byte/pixel/near component exclusion before split. Shared real_test scored once; unseen/seen tests are AI-only manifests.',
                  checkpoint_rule='max validation AUROC, tie lower BCE',threshold_rule='lowest threshold with empirical validation real FPR <= 0.01, ties conservative',
                  final_tests_opened_for_model_selection=False)
    write_json(OUTPUT/'split_config.json',metadata)
    write_json(AUDIT/'audit_summary.json',metadata)
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__': main()
