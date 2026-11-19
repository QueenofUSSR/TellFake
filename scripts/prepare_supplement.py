"""Freeze new holdouts independently, without changing any v3 fingerprint."""
import csv,hashlib,io,json,random,sys
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from genimage_detector.data import load_manifests,read_csv,write_csv,write_json,file_hash

ROOT=Path('/mnt/ssd/data/datasets')
MID=ROOT/'GenImage/official_v3/Midjourney_subset'
OUT=Path('data/manifests/v3_supplement_20261113')
AUDIT=Path('/mnt/ssd/data/logs/arin5201/supplement_20261113')
FIELDS=('path','label','generator','split','source_split','width','height','format','sha256','pixel_sha256','source','original_id','group_id','dhash')

def inspect(item):
 path,generator,expected=item
 content=path.read_bytes();sha=hashlib.sha256(content).hexdigest()
 if expected and sha!=expected:raise ValueError('Transfer manifest hash mismatch '+str(path))
 with Image.open(io.BytesIO(content)) as im:
  im.load();fmt=im.format;rgb=ImageOps.exif_transpose(im).convert('RGB')
  pixel=hashlib.sha256(f'{rgb.width}x{rgb.height}:'.encode()+rgb.tobytes()).hexdigest()
  small=np.array(rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS));dh=int.from_bytes(np.packbits(small[:,1:]>small[:,:-1]).tobytes(),'big')
 original=generator+'/'+path.stem
 return dict(path=str(path.relative_to(ROOT)),label=int(generator=='Midjourney'),generator='Midjourney' if generator=='Midjourney' else 'Real',split='midjourney_test' if generator=='Midjourney' else 'external_real_test',source_split='val',width=rgb.width,height=rgb.height,format=fmt,sha256=sha,pixel_sha256=pixel,source=generator,original_id=original,group_id=hashlib.sha256(original.encode()).hexdigest(),dhash=f'{dh:016x}')

def main():
 if OUT.exists():raise FileExistsError(OUT)
 AUDIT.mkdir(parents=True,exist_ok=True)
 base,fp=load_manifests('data/manifests/v3');previous=[r for rs in base.values() for r in rs]
 old_file={r['sha256'] for r in previous};old_pixel={r['pixel_sha256'] for r in previous}
 for p in Path('data/manifests/v2').glob('*.csv'):
  for r in read_csv(p):
   if r.get('sha256'):old_file.add(r['sha256'])
   if r.get('pixel_sha256'):old_pixel.add(r['pixel_sha256'])
 items=[];manifests={}
 for manifest in sorted(MID.glob('batch_*/manifest.json')):
  manifests[str(manifest)]=file_hash(manifest)
  for r in json.loads(manifest.read_text()):
   if '/val/ai/' in r['path']:items.append((manifest.parent/r['path'],'Midjourney',r['sha256']))
 items += [(p,'COCO2017',None) for p in sorted((ROOT/'val2017').glob('*.jpg'))]
 rows=[]
 with ThreadPoolExecutor(max_workers=8) as ex:
  for i,r in enumerate(ex.map(inspect,items),1):
   rows.append(r)
   if i%1000==0:print('Decoded',i,'/',len(items),flush=True)
 write_csv(AUDIT/'decoded_candidates.csv',rows,FIELDS)
 # Base-selected near hashes and all selected/historical exact hashes are barred.
 bands=[defaultdict(list) for _ in range(5)];hashes=[];paths=[]
 def add(r):
  i=len(hashes);h=int(r['dhash'],16);hashes.append(h);paths.append(r['path'])
  for b in range(5):bands[b][(h>>(b*13))&8191].append(i)
 for r in previous:add(r)
 rng=random.Random(42);rng.shuffle(rows);kept=defaultdict(list);excluded=[]
 seen_original={r['original_id'] for r in previous}
 for r in rows:
  reason=None;matches=[]
  if r['sha256'] in old_file or r['pixel_sha256'] in old_pixel:reason='exact_overlap'
  elif r['original_id'] in seen_original:reason='original_overlap'
  else:
   h=int(r['dhash'],16);possible=set()
   for b in range(5):possible.update(bands[b][(h>>(b*13))&8191])
   matches=[paths[j] for j in sorted(possible) if (h^hashes[j]).bit_count()<=4]
   if matches:reason='near_candidate_conservative_exclusion'
  if reason:excluded.append(dict(path=r['path'],reason=reason,matches='|'.join(matches)))
  else:
   kept[r['split']].append(r);old_file.add(r['sha256']);old_pixel.add(r['pixel_sha256']);seen_original.add(r['original_id']);add(r)
 if len(kept['midjourney_test'])<2000:raise ValueError('Not enough independent Midjourney samples')
 kept['midjourney_test']=kept['midjourney_test'][:2000]
 OUT.mkdir(parents=True)
 for name,rs in kept.items():write_csv(OUT/(name+'.csv'),rs,FIELDS)
 write_csv(AUDIT/'excluded.csv',excluded,('path','reason','matches'))
 metadata=dict(schema_version=1,kind='frozen_v3_supplement',seed=42,data_root=str(ROOT),base_manifest_sha256=fp,
  counts={n:len(rs) for n,rs in kept.items()},manifest_sha256={n:file_hash(OUT/(n+'.csv')) for n in kept},
  source_manifests_sha256=manifests,original_decoded=len(rows),exclusions=dict(Counter(r['reason'] for r in excluded)),
  distributions={n:dict(Counter(f"{r['format']}/{r['width']}x{r['height']}" for r in rs)) for n,rs in kept.items()},
  provenance={'Midjourney':'Original GenImage val/ai extracted remotely; local hashes verified against per-batch transfer manifests. Not previously fitted or used for tuning.',
              'COCO2017':'User supplied val2017; 5000 named JPEG files. Official COCO val2017 is an external natural-image dataset, not per-image camera-authenticity certification. Archive/annotation identity not independently certified here.',
              'COCO_terms':'https://cocodataset.org/#termsofuse; individual image licenses apply; no redistribution of source pictures'},
  limitations=['dHash<=4 is a heuristic; conservative exclusions can reject unrelated images','Pixel and near comparisons against v3 selected images; exact comparisons also include v2. Not a full-reserve near audit.','COCO may contain edited/composite/artwork images; not a camera-only certified set.','Do not silently replace exclusions to claim 5000 independent external images.'],
  policy='All supplemental images are final-only. Existing v3 train/val/dev/tests immutable. Midjourney pairs with the existing shared real_test, COCO FPR is reported separately. No checkpoint/threshold/method selection on supplements.',final_predictions_run=False)
 write_json(OUT/'supplement_config.json',metadata);write_json(AUDIT/'summary.json',metadata)
 assert load_manifests('data/manifests/v3')[1]==fp
 print('Frozen supplement:',metadata['counts'],'excluded',metadata['exclusions'],flush=True)

if __name__=='__main__':main()
