"""Serial mirror download; verify, extract and remove only verified archives."""
import hashlib,json,shutil,subprocess,urllib.request,zipfile,os,time
from pathlib import Path
from PIL import Image
BASE=Path('/mnt/ssd/data/downloads/arin5201/genimage_v3')
OUT=Path('/mnt/ssd/data/datasets/GenImage/official_v3')
LOG=Path('/mnt/ssd/data/logs/arin5201/downloads_v3')
REPO='ENSTA-U2IS/GenImage'
for p in (BASE,OUT,LOG):p.mkdir(parents=True,exist_ok=True)
def api(path):
 return json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/'+REPO+path,timeout=60))
rev=api('')['sha']
files=api('/tree/'+rev+'/BigGAN?expand=true')
files=[f for f in files if f['type']=='file']
(LOG/'biggan_mirror_manifest.json').write_text(json.dumps({'repo':REPO,'revision':rev,'files':files},indent=2))
size=sum(f['size'] for f in files)
if shutil.disk_usage(BASE).free < size*3+50*1024**3:raise RuntimeError('Insufficient space')
for f in sorted(files,key=lambda x:x['path']):
 p=BASE/Path(f['path']).name
 if not p.exists():
  partial=Path(str(p)+'.part')
  url='https://huggingface.co/datasets/'+REPO+'/resolve/'+rev+'/'+f['path']+'?download=true'
  print('Downloading',f['path'],f['size'],flush=True)
  for attempt in range(20):
   result=subprocess.run(['curl','-fL','--connect-timeout','30','--speed-limit','1024','--speed-time','90','-C','-','-o',str(partial),url])
   if result.returncode==0:break
   print('Retry',attempt+1,'exit',result.returncode,flush=True)
   time.sleep(15)
  else:raise RuntimeError('Download failed after 20 resumable attempts')
  partial.rename(p)
 if p.stat().st_size!=f['size']:raise RuntimeError('Size mismatch '+str(p))
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
 if h.hexdigest()!=f['lfs']['oid']:raise RuntimeError('SHA256 mismatch '+str(p))
 print('Verified',p.name,flush=True)
merged=BASE/'biggan_merged.zip'
original=next(BASE/Path(f['path']).name for f in files if f['path'].endswith('.zip'))
if not merged.exists():subprocess.run(['zip','-s','0',str(original),'--out',str(merged)],check=True)
staging=OUT/'BigGAN.staging'
final=OUT/'BigGAN'
if final.exists():raise RuntimeError('Destination already exists; manual reconciliation required')
with zipfile.ZipFile(merged) as z:
 infos=z.infolist(); expanded=sum(i.file_size for i in infos)
 if shutil.disk_usage(BASE).free<expanded+50*1024**3:raise RuntimeError('Insufficient extraction space')
 for i in infos:
  q=Path(i.filename)
  if q.is_absolute() or '..' in q.parts or ((i.external_attr>>16)&0o170000)==0o120000:raise RuntimeError('Unsafe archive entry')
 bad=z.testzip()
 if bad:raise RuntimeError('Archive CRC failure '+bad)
 z.extractall(staging)
 count=0
 for i in infos:
  if i.is_dir():continue
  p=staging/i.filename
  if p.stat().st_size!=i.file_size:raise RuntimeError('Extraction mismatch')
  if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp','.bmp'):
   with Image.open(p) as im:im.load()
   count+=1
staging.rename(final)
record={'status':'verified_extracted','images':count,'destination':str(final),'revision':rev,'archive_bytes':size}
(LOG/'biggan_complete.json').write_text(json.dumps(record,indent=2))
for f in files:(BASE/Path(f['path']).name).unlink()
merged.unlink()
print('Completed',json.dumps(record),flush=True)
