import json,hashlib,sqlite3,zipfile,subprocess,shutil,os,zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
ROOT=Path('/mnt/ssd/data/datasets/GenImage/official_v3'); LOG=Path('/mnt/ssd/data/logs/arin5201/reserve_20261114');LOG.mkdir(parents=True,exist_ok=True)
def emit(*args):print(*args,flush=True)
def audit(name,paths):
 def check(p):
  try:
   with Image.open(p) as im:im.load()
   return None
  except Exception as e:return p,str(e)
 bad=[]
 with ThreadPoolExecutor(max_workers=8) as pool:
  for start in range(0,len(paths),2000):
   for result in pool.map(check,paths[start:start+2000]):
    if result:bad.append(result)
   if start%20000==0:emit(name,'decoded',min(start+2000,len(paths)),'bad',len(bad))
 for p,error in bad:
  rel=p.relative_to(ROOT/(name+'.staging'));dest=ROOT/'quarantine'/name/rel;dest.parent.mkdir(parents=True,exist_ok=True);p.rename(dest)
 (LOG/(name+'_bad.json')).write_text(json.dumps([{'path':str(p),'error':e} for p,e in bad],indent=2))
 (ROOT/(name+'.staging')).rename(ROOT/name)
 emit(name,'audit complete',len(paths)-len(bad),'valid',len(bad),'quarantined')
def images(root):return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp'))
if not (ROOT/'SD14').exists():audit('SD14',images(ROOT/'SD14.staging'))
arc=Path('/mnt/ssd/data/downloads/arin5201/lan_transfer/ADM'); merged=arc/'adm_merged.zip'
if not (ROOT/'ADM').exists():
 volumes=sorted(p for p in arc.iterdir() if p.name.startswith('imagenet_ai_0508_adm.') and p.is_file())
 if not merged.exists():subprocess.run(['zip','-q','-s','0',str(arc/'imagenet_ai_0508_adm.zip'),'--out',str(merged)],check=True)
 with zipfile.ZipFile(merged) as z:
  infos=z.infolist();required=sum(i.file_size for i in infos)
  if shutil.disk_usage(ROOT).free<required+50*1024**3:raise RuntimeError('Insufficient space')
  for i in infos:
   p=Path(i.filename)
   if p.is_absolute() or '..' in p.parts or ((i.external_attr>>16)&0o170000)==0o120000:raise RuntimeError('Unsafe path')
  emit('ADM extracting',len(infos),'entries')
  z.extractall(ROOT/'ADM.staging') # zipfile verifies CRC during extraction
  for i in infos:
   if not i.is_dir() and (ROOT/'ADM.staging'/i.filename).stat().st_size!=i.file_size:raise RuntimeError('Extraction mismatch')
 audit('ADM',images(ROOT/'ADM.staging'))
 (LOG/'ADM_archives_removed.json').write_text(json.dumps([{'path':str(p),'size':p.stat().st_size} for p in volumes+[merged]],indent=2))
 for p in volumes+[merged]:p.unlink()
 emit('ADM archives removed')
# Inventory content hashes, bounded batches. Files remain in their original directories.
db=sqlite3.connect(LOG/'retention.sqlite');db.execute('CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, source TEXT, label TEXT, sha TEXT, size INTEGER, priority TEXT, keep INTEGER DEFAULT 0)')
quotas={'BigGAN':75000,'SD15':75000,'SD14':69613,'ADM':69613,'GLIDE':69613,'VQDM':69612,'Midjourney_subset':21549}
def hashed(item):
 source,p=item;rel=str(p.relative_to(ROOT));label='real' if 'nature' in p.parts else 'ai'
 h=hashlib.sha256(p.read_bytes()).hexdigest();key=('0' if 'val' in p.parts else '1')+hashlib.sha256(('42:'+rel).encode()).hexdigest()
 return rel,source,label,h,p.stat().st_size,key
for source in quotas:
 paths=images(ROOT/source)
 with ThreadPoolExecutor(max_workers=4) as pool:
  for start in range(0,len(paths),2000):
   db.executemany('INSERT OR REPLACE INTO files(path,source,label,sha,size,priority) VALUES(?,?,?,?,?,?)',pool.map(hashed,[(source,p) for p in paths[start:start+2000]]));db.commit()
   if start%50000==0:emit('Hash inventory',source,min(start+2000,len(paths)),len(paths))
conflicts=db.execute('SELECT sha FROM files GROUP BY sha HAVING COUNT(DISTINCT label)>1').fetchall()
if conflicts:raise RuntimeError('Cross-label content conflicts; no retention deletion performed')
db.execute('UPDATE files SET keep=0');used=set();kept={}
for source,quota in quotas.items():
 n=0
 for path,sha in db.execute("SELECT path,sha FROM files WHERE source=? AND label='ai' ORDER BY priority",(source,)):
  if sha in used:continue
  used.add(sha);db.execute('UPDATE files SET keep=1 WHERE path=?',(path,));n+=1
  if n==quota:break
 kept[source]=n
# Fill any generator's content-duplicate shortfall from unused AI, retaining source diversity already selected.
missing=450000-sum(kept.values())
if missing>0:
 for path,source,sha in db.execute("SELECT path,source,sha FROM files WHERE label='ai' AND keep=0 ORDER BY priority"):
  if sha in used:continue
  used.add(sha);db.execute('UPDATE files SET keep=1 WHERE path=?',(path,));kept[source]+=1;missing-=1
  if not missing:break
n=0
for path,sha in db.execute("SELECT path,sha FROM files WHERE label='real' ORDER BY priority"):
 if sha in used:continue
 used.add(sha);db.execute('UPDATE files SET keep=1 WHERE path=?',(path,));n+=1
 if n==444000:break
if missing or n<444000:raise RuntimeError('Not enough unique images; no deletion performed')
db.commit()
# Exact per-file keep/delete manifests written before deletion.
import csv
for flag,name in [(1,'retained.csv'),(0,'removed.csv')]:
 with (LOG/name).open('w') as out:
  w=csv.writer(out);w.writerow(['path','source','label','sha256','bytes']);w.writerows(db.execute('SELECT path,source,label,sha,size FROM files WHERE keep=? ORDER BY path',(flag,)))
removed=0;freed=0
for rel,size in db.execute('SELECT path,size FROM files WHERE keep=0'):
 p=ROOT/rel
 if p.is_symlink() or not p.is_relative_to(ROOT):raise RuntimeError('Unexpected path')
 p.unlink();removed+=1;freed+=size
 if removed%50000==0:emit('Removed excess',removed)
for rel,size in db.execute('SELECT path,size FROM files WHERE keep=1'):
 if (ROOT/rel).stat().st_size!=size:raise RuntimeError('Retention verification failure')
result={'status':'complete','unique_images':894000,'ai':450000,'real':444000,'ai_by_source':kept,'removed_images':removed,'released_image_bytes':freed,'dedup':'SHA256 of original file bytes; perceptual duplicates not assessed','free_GiB':shutil.disk_usage(ROOT).free/1024**3}
(LOG/'complete.json').write_text(json.dumps(result,indent=2));emit('COMPLETE',result)
