import json,shutil,subprocess,time
from pathlib import Path
from PIL import Image
root=Path('/mnt/ssd/data/datasets/GenImage/official_v3')
staging=root/'VQDM.staging'
logs=Path('/mnt/ssd/data/logs/arin5201/downloads_v3')
archives=Path('/mnt/ssd/data/downloads/arin5201/lan_transfer/VQDM')
import urllib.request
repo='jzousz/GenImage'
rev=json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/'+repo))['sha']
files=json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/'+repo+'/tree/'+rev+'/VQDM?expand=true'))
manifest={'revision':rev,'files':[f for f in files if f['type']=='file']}
(logs/'vqdm_mirror_manifest.json').write_text(json.dumps(manifest,indent=2))
import hashlib,zipfile
for f in manifest['files']:
 p=archives/Path(f['path']).name
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
 if h.hexdigest()!=f['lfs']['oid']:raise RuntimeError('Hash mismatch '+str(p))
merged=archives/'vqdm_merged.zip'
if not merged.exists():
 original=next(archives/Path(f['path']).name for f in manifest['files'] if f['path'].endswith('.zip'))
 subprocess.run(['zip','-q','-s','0',str(original),'--out',str(merged)],check=True)
with zipfile.ZipFile(merged) as z:
 expanded=sum(i.file_size for i in z.infolist())
 if shutil.disk_usage(root).free < expanded+50*1024**3:raise RuntimeError('Insufficient extraction space')
 for i in z.infolist():
  q=Path(i.filename)
  if q.is_absolute() or '..' in q.parts or ((i.external_attr>>16)&0o170000)==0o120000:raise RuntimeError('Unsafe archive entry')
 badentry=z.testzip()
 if badentry:raise RuntimeError('CRC failure '+badentry)
 z.extractall(staging)
 for i in z.infolist():
  if not i.is_dir() and (staging/i.filename).stat().st_size!=i.file_size:raise RuntimeError('Missing extraction '+i.filename)
bad=[]; good=0
for p in staging.rglob('*'):
 if p.suffix.lower() not in ('.png','.jpg','.jpeg','.bmp','.webp') or not p.is_file():continue
 try:
  with Image.open(p) as im:im.load()
  good+=1
 except Exception as e:bad.append({'path':str(p.relative_to(staging)),'error':str(e)})
 if (good+len(bad))%10000==0:print('Checked',good+len(bad),'bad',len(bad),flush=True)
(logs/'vqdm_decode_audit.json').write_text(json.dumps({'good':good,'bad':bad},indent=2))
# Preserve source anomalies separately; do not include them in training.
for item in bad:
 p=staging/item['path']; q=root/'quarantine'/'VQDM'/item['path'];q.parent.mkdir(parents=True,exist_ok=True);p.rename(q)
staging.rename(root/'VQDM')
(logs/'vqdm_complete.json').write_text(json.dumps({'status':'verified_with_quarantined_source_errors','good_images':good,'quarantined':len(bad),'destination':str(root/'VQDM')},indent=2))
for f in manifest['files']:(archives/Path(f['path']).name).unlink()
(archives/'vqdm_merged.zip').unlink()
print('VQDM audited; source anomalies preserved in quarantine',len(bad),flush=True)
if shutil.disk_usage(root).free < 130*1024**3:raise RuntimeError('Insufficient reserve for VQDM')
dest=Path('/mnt/ssd/data/downloads/arin5201/lan_transfer/GLIDE');dest.mkdir(parents=True,exist_ok=True)
with (logs/'lan_glide_transfer.log').open('a') as log:
 subprocess.run(['rsync','-rt','--partial','--checksum','--info=progress2','-e','ssh -S /tmp/arin5201-transfer-ssh','tom@192.168.60.1:/home/tom/文档/GenImage/glide/',str(dest)+'/'],stdout=log,stderr=subprocess.STDOUT,check=True)
print('GLIDE transfer completed',flush=True)
