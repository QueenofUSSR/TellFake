"""P0 bounded candidate audit. Source images and historical manifests are read-only.

Inventory covers the retained reserve. Every prospective candidate is fully decoded;
unused reserve is NOT claimed to have passed this round's pixel/near-duplicate audit.
"""
import csv, hashlib, io, json, os, random, re, sqlite3, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, JpegImagePlugin
PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path('/mnt/ssd/data/datasets/GenImage/official_v3')
OUT=PROJECT/'reproduction'
AUDIT=Path('/mnt/ssd/data/logs/arin5201/reproduction/p0_20261114')
RESERVE=Path('/mnt/ssd/data/logs/arin5201/reserve_20261114/retained.csv')
SEED=20261114
FIELDS=['path','label','generator','source','source_split','split','original_id','group_id','sha256','pixel_sha256','width','height','format','dhash','jpeg_subsampling','jpeg_quality_estimate','quantization_sha256']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def csvwrite(p,rows,fields=FIELDS):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def inspect(r):
 try:
  data=Path(r['path']).read_bytes(); h=hashlib.sha256(data).hexdigest()
  if r.get('sha256') and h!=r['sha256']:raise ValueError('file hash changed')
  with Image.open(io.BytesIO(data)) as im:
   im.load();fmt=im.format; subs=JpegImagePlugin.get_sampling(im) if fmt=='JPEG' else -1
   quant=getattr(im,'quantization',None); qhash=hashlib.sha256(json.dumps(quant,sort_keys=True).encode()).hexdigest() if quant else ''
   # Only claim a quality estimate when the first quantization table matches IJG.
   base=[16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99]
   quality=''
   if quant and 0 in quant:
    for q in range(1,101):
     scale=5000//q if q<50 else 200-2*q
     if [max(1,min(255,(v*scale+50)//100)) for v in base]==quant[0]:quality=q;break
   rgb=ImageOps.exif_transpose(im).convert('RGB');px=hashlib.sha256(f'{rgb.width}x{rgb.height}:'.encode()+rgb.tobytes()).hexdigest()
   a=np.array(rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS));dh=np.packbits(a[:,1:]>a[:,:-1]).tobytes().hex()
  return dict(r,sha256=h,pixel_sha256=px,width=rgb.width,height=rgb.height,format=fmt,dhash=dh,jpeg_subsampling=subs,jpeg_quality_estimate=quality,quantization_sha256=qhash,error='')
 except Exception as e:return dict(r,error=f'{type(e).__name__}: {e}')
def main():
 if (OUT/'P0_ACCEPTANCE.json').exists():
  raise RuntimeError('P0 version already accepted. Validate existing manifests or create a NEW version; never overwrite frozen splits.')
 AUDIT.mkdir(parents=True,exist_ok=True)
 inventory=Counter();bytes_=Counter();pools=defaultdict(list);path_seen=set();missing=[]
 with RESERVE.open() as f:
  for r in csv.DictReader(f):
   p=ROOT/r['path'];parts=p.parts;sp='train' if 'train' in parts else 'val' if 'val' in parts else 'unknown'
   key=f"{r['source']}/{sp}/{r['label']}";inventory[key]+=1;bytes_[key]+=int(r['bytes'])
   if not p.is_file():missing.append(str(p));continue
   label=int(r['label']=='ai');gen=r['source'] if label else 'Real'
   oid=('ImageNet/' if not label else gen+'/')+p.stem
   row=dict(path=str(p),label=label,generator=gen,source=r['source'],source_split=sp,original_id=oid,sha256=r['sha256'],split='')
   pools[gen,sp].append(row)
 # Exclude ALL previously used images (including development), not merely final tests.
 historical={}
 for f in sorted((PROJECT/'data/manifests').glob('*/*.csv')):
  if f.stem in ('unused','duplicates'):continue
  with f.open() as stream:
   for r in csv.DictReader(stream):
    if not r.get('path'):continue
    p=Path(r['path']);legacy_root=Path('/mnt/ssd/data/datasets/GenImage/standard_v1') if f.parent.name=='v2' else ROOT
    p=p if p.is_absolute() else legacy_root/p
    if not p.is_file():
     # v2 absolute locations can be stale; hashes/identities still block selection below.
     pass
    gen=r.get('generator','');oid=r.get('original_id') or ('ImageNet/' if str(r.get('label'))=='0' else gen+'/')+p.stem
    historical[str(p)]=dict(r,path=str(p),original_id=oid,historical=True)
 old_oid={r['original_id'] for r in historical.values()};old_sha={r.get('sha256') for r in historical.values()}-{None,''}
 chosen=[];rng=random.Random(SEED)
 # Prospective scope: U S0-S3, R SD14/BigGAN/ADM S1-S2, fixed holdouts.
 limits={('Real','train'):100000,('SD14','train'):60000,('BigGAN','train'):50000,('ADM','train'):22000}
 for key,pool in sorted(pools.items()):
  if key[1]!='val' and key not in limits:continue
  pool.sort(key=lambda r:r['path']);rng.shuffle(pool);seen=set();eligible=[]
  for r in pool:
   if r['original_id'] in seen or r['original_id'] in old_oid or r['sha256'] in old_sha:continue
   seen.add(r['original_id']);eligible.append(r)
  chosen.extend(eligible[:limits.get(key,len(eligible))])
 # Historic dHash/pixel keys are read from frozen manifests; decode older entries lacking them.
 historical_decode=[r for r in historical.values() if not r.get('pixel_sha256') or not r.get('dhash')]
 historical_decode=[r for r in historical_decode if Path(r['path']).is_file()]
 selected=chosen+historical_decode
 csvwrite(AUDIT/'candidate_input.csv',selected,['path','label','generator','source','source_split','original_id','sha256','historical'])
 dump(OUT/'data_inventory.json',dict(status='auditing',root=str(ROOT),inventory_counts=dict(inventory),inventory_bytes=dict(bytes_),retained_total=sum(inventory.values()),missing=missing,candidate_count=len(chosen),historical_decode_count=len(historical_decode),scope='All retained paths counted and existence-checked. Full decode/pixel/dHash audit covers candidate_input.csv plus historical hashes; unused reserve not pixel-certified.',reserve_csv_sha256=sha(RESERVE)))
 if missing:raise RuntimeError(f'{len(missing)} retained files missing')
 db=sqlite3.connect(AUDIT/'decoded.sqlite');db.execute('CREATE TABLE IF NOT EXISTS decoded(path TEXT PRIMARY KEY, sha TEXT, size INTEGER, mtime INTEGER, row_json TEXT)')
 cache={p:(h,size,mt,j) for p,h,size,mt,j in db.execute('SELECT * FROM decoded')};rows=[];bad=[];start=time.time()
 with ProcessPoolExecutor(max_workers=12) as ex:
  for offset in range(0,len(selected),1024):
   chunk=selected[offset:offset+1024];pending=[]
   for r in chunk:
    st=Path(r['path']).stat();hit=cache.get(r['path'])
    if hit and hit[1:3]==(st.st_size,st.st_mtime_ns) and (not r.get('sha256') or hit[0]==r['sha256']):
     item=dict(r,**{k:v for k,v in json.loads(hit[3]).items() if k not in r});(bad if item.get('error') else rows).append(item)
    else:pending.append(r)
   for r in ex.map(inspect,pending,chunksize=16):
    st=Path(r['path']).stat();db.execute('INSERT OR REPLACE INTO decoded VALUES(?,?,?,?,?)',(r['path'],r.get('sha256',''),st.st_size,st.st_mtime_ns,json.dumps(r)))
    (bad if r.get('error') else rows).append(r)
   db.commit()
   if offset%10240==0:print('Decoded',min(offset+1024,len(selected)),'/',len(selected),'seconds',round(time.time()-start),flush=True)
 dump(AUDIT/'bad_images.json',bad)
 for r in rows:
  if r.get('historical'):historical[r['path']]=r
 candidates=[r for r in rows if not r.get('historical')]
 hist=[r for r in historical.values() if r.get('pixel_sha256') and r.get('dhash')]
 # Any historic hash with incomplete metadata remains blocked by original/file identity.
 allrows=hist+candidates;n=len(allrows);parent=list(range(n));bands=[defaultdict(list) for _ in range(5)];exact={};ident={};hashrep={};pairs=[]
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(i,j):
  a,b=find(i),find(j)
  if a!=b:parent[max(a,b)]=min(a,b)
 for i,r in enumerate(allrows):
  for mapping,key in [(exact,r['pixel_sha256']),(ident,r['original_id'])]:
   if key in mapping:union(i,mapping[key])
   else:mapping[key]=i
  h=int(r['dhash'],16)
  if h in hashrep:
   j=hashrep[h];union(i,j)
   if len(pairs)<10000:pairs.append({'path_a':allrows[j]['path'],'path_b':r['path'],'distance':0})
   continue
  possible=set()
  for b in range(5):possible.update(bands[b][(h>>(13*b))&8191])
  for j in possible:
   dist=(h^int(allrows[j]['dhash'],16)).bit_count()
   if dist<=4:
    union(i,j)
    if len(pairs)<10000:pairs.append({'path_a':allrows[j]['path'],'path_b':r['path'],'distance':dist})
  hashrep[h]=i
  for b in range(5):bands[b][(h>>(13*b))&8191].append(i)
  if i%20000==0:print('Near retrieval',i,'/',n,flush=True)
 groups=defaultdict(list)
 for i,r in enumerate(allrows):groups[find(i)].append(r)
 eligible=defaultdict(list);excluded=[]
 for group in groups.values():
  reason=None
  if any(r['path'] in historical for r in group):reason='historical_identity_or_pixel_or_dhash_component'
  elif len({str(r['label']) for r in group})>1:reason='mixed_label_component'
  elif len({r['generator'] for r in group if int(r['label'])})>1:reason='mixed_generator_component'
  group.sort(key=lambda r:(r.get('source_split')!='val',r['path']))
  if reason:
   excluded.extend(dict(path=r['path'],reason=reason) for r in group if r['path'] not in historical);continue
  r=group[0];r['group_id']=hashlib.sha256('\n'.join(sorted({x['original_id'] for x in group})).encode()).hexdigest()
  eligible[r['generator'],r['source_split']].append(r)
  excluded.extend(dict(path=x['path'],reason='duplicate_component_representative_only') for x in group[1:])
 csvwrite(AUDIT/'near_duplicate_examples.csv',pairs,['path_a','path_b','distance']);csvwrite(AUDIT/'exclusions.csv',excluded,['path','reason'])
 for key,pool in sorted(eligible.items()):
  pool.sort(key=lambda r:r['path']);rng.shuffle(pool)
  strata=defaultdict(list)
  for row in pool:
   match=re.match(r'^(n[0-9]{8}|[0-9]{1,3})_',Path(row['path']).stem)
   strata[match.group(1) if match else 'unknown'].append(row)
  keys=sorted(strata);rng.shuffle(keys);ordered=[]
  while any(strata.values()):
   for k in keys:
    if strata[k]:ordered.append(strata[k].pop())
  eligible[key]=ordered
 availability={f'{a}/{b}':len(v) for (a,b),v in eligible.items()};dump(AUDIT/'availability.json',availability)
 manifests=OUT/'manifests/v1_20261114';manifests.mkdir(parents=True,exist_ok=True)
 splits={}
 def take(name,allocation):
  out=[]
  for key,count in allocation.items():
   if len(eligible[key])<count:raise RuntimeError(f'Insufficient {key}: {len(eligible[key])}/{count}; see availability.json')
   out.extend(dict(r,split=name) for r in eligible[key][:count]);del eligible[key][:count]
  splits[name]=out
 take('val',{('Real','train'):5000,('SD14','train'):2500,('BigGAN','train'):2500})
 take('dev',{('Real','train'):4000,('SD14','train'):2000,('BigGAN','train'):2000})
 take('train_S3',{('Real','train'):64000,('SD14','train'):32000,('BigGAN','train'):32000})
 for name,count in [('train_S2',64000),('train_S1',16000),('train_S0',4000)]:
  group=defaultdict(list)
  for r in splits['train_S3']:group[r['generator']].append(r)
  splits[name]=[dict(r,split=name) for g,k in [('Real',count//2),('SD14',count//4),('BigGAN',count//4)] for r in group[g][:k]]
 take('test_real',{('Real','val'):5000})
 # Holdout targets are 2k-5k; use 2k per AI source to preserve unseen reserves.
 for gen in ['SD14','BigGAN','ADM','GLIDE','SD15','VQDM','Midjourney_subset']:take('test_'+gen,{(gen,'val'):2000})
 # R-only extras are isolated from U and all holdouts; ADM never enters U training.
 take('R_ADM_train_pool',{('ADM','train'):16000})
 take('R_SD14_train_extra',{('SD14','train'):16000})
 for name,rs in splits.items():csvwrite(manifests/(name+'.csv'),rs)
 distributions={name:dict(Counter(f"{r['generator']}/{r['format']}/{r['width']}x{r['height']}/sub{r['jpeg_subsampling']}/q{r['jpeg_quality_estimate']}" for r in rs)) for name,rs in splits.items()}
 dump(AUDIT/'encoding_distributions.json',distributions)
 config=dict(stratification='Source/label balanced; round-robin filename class prefixes where recoverable (ImageNet synset or generator class index); unknown classes not invented; no claim of shared cross-source semantic class alignment.',schema_version=1,seed=SEED,data_root=str(ROOT),seen=['SD14','BigGAN'],primary_unseen=['ADM','GLIDE'],same_family=['SD15'],extended=['VQDM','Midjourney_subset'],counts={k:dict(Counter(r['generator'] for r in v)) for k,v in splits.items()},manifest_sha256={k:sha(manifests/(k+'.csv')) for k in splits},audit_db=str(AUDIT/'decoded.sqlite'),audit_scope='candidate_input.csv; not whole retained reserve',near_duplicate_algorithm='EXIF-normalized RGB pixel SHA256, ImageNet filename identity, 64-bit grayscale dHash Hamming <=4; exhaustive five-band retrieval within audited candidates and historical keys; one representative per connected component. Not a guarantee against crops or all perceptual duplicates.',historical_manifest_sha256={str(p.relative_to(PROJECT)):sha(p) for p in (PROJECT/'data/manifests').glob('*/*.csv') if p.stem not in ('unused','duplicates')},test_policy='All prior used manifests excluded, official val only for confirmation; same domains as historical research; no new test scores inspected.',derivative_policy='Derivatives must inherit original_id, group_id and split; no independently split crops/JPEG/VAE twins.',availability_before_allocation=availability)
 dump(manifests/'protocol.json',config)
 inv=json.loads((OUT/'data_inventory.json').read_text());inv.update(status='audited_candidates_frozen_pending_validation',decoded_candidates=len(candidates),bad_count=len(bad),excluded_count=len(excluded),historical_count=len(historical),manifest_dir=str(manifests),audit_dir=str(AUDIT));dump(OUT/'data_inventory.json',inv)
 print('FROZEN',config['counts'],flush=True)
if __name__=='__main__':main()
