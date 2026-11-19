from pathlib import Path
import hashlib,json,sqlite3
from concurrent.futures import ThreadPoolExecutor
root=Path('/mnt/ssd/data/datasets/GenImage/official_v3')
log=Path('/mnt/ssd/data/logs/arin5201/downloads_v3')
db=sqlite3.connect(log/'inventory_hashes.sqlite');db.execute('CREATE TABLE IF NOT EXISTS images(path TEXT PRIMARY KEY, sha256 TEXT, label TEXT)')
paths=(p for name in ['BigGAN','SD15','VQDM','GLIDE'] for p in (root/name).rglob('*') if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.webp'))
def hashfile(p):
 return str(p.relative_to(root)),hashlib.sha256(p.read_bytes()).hexdigest(),'real' if 'nature' in p.parts else 'ai'
with ThreadPoolExecutor(max_workers=4) as ex:
 for n,row in enumerate(ex.map(hashfile,paths),1):
  db.execute('INSERT OR REPLACE INTO images VALUES(?,?,?)',row)
  if n%10000==0:db.commit();print('Hashed',n,flush=True)
db.commit()
result={'counts':list(db.execute('SELECT label,COUNT(*),COUNT(DISTINCT sha256) FROM images GROUP BY label')),'cross_label_conflicts':db.execute('SELECT COUNT(*) FROM (SELECT sha256 FROM images GROUP BY sha256 HAVING COUNT(DISTINCT label)>1)').fetchone()[0]}
(log/'inventory_dedup.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
