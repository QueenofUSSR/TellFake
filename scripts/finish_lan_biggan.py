import json,shutil,subprocess,time
from pathlib import Path
from PIL import Image
root=Path('/mnt/ssd/data/datasets/GenImage/official_v3')
staging=root/'BigGAN.staging'
logs=Path('/mnt/ssd/data/logs/arin5201/downloads_v3')
archives=Path('/mnt/ssd/data/downloads/arin5201/lan_transfer/BigGAN')
manifest=json.loads((logs/'biggan_mirror_manifest.json').read_text())
import hashlib,zipfile
for f in manifest['files']:
 p=archives/Path(f['path']).name
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
 if h.hexdigest()!=f['lfs']['oid']:raise RuntimeError('Hash mismatch '+str(p))
with zipfile.ZipFile(archives/'biggan_merged.zip') as z:
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
(logs/'biggan_decode_audit.json').write_text(json.dumps({'good':good,'bad':bad},indent=2))
# Preserve source anomalies separately; do not include them in training.
for item in bad:
 p=staging/item['path']; q=root/'quarantine'/'BigGAN'/item['path'];q.parent.mkdir(parents=True,exist_ok=True);p.rename(q)
staging.rename(root/'BigGAN')
(logs/'biggan_complete.json').write_text(json.dumps({'status':'verified_with_quarantined_source_errors','good_images':good,'quarantined':len(bad),'destination':str(root/'BigGAN')},indent=2))
for f in manifest['files']:(archives/Path(f['path']).name).unlink()
(archives/'biggan_merged.zip').unlink()
print('BigGAN audited; source anomalies preserved in quarantine',len(bad),flush=True)
if shutil.disk_usage(root).free < 300*1024**3:raise RuntimeError('Insufficient reserve for SD15')
dest=Path('/mnt/ssd/data/downloads/arin5201/lan_transfer/SD15');dest.mkdir(parents=True,exist_ok=True)
with (logs/'lan_sd15_transfer.log').open('a') as log:
 subprocess.run(['rsync','-rt','--partial','--checksum','--info=progress2','-e','ssh -S /tmp/arin5201-transfer-ssh','tom@192.168.60.1:/home/tom/文档/GenImage/stable_diffusion_v_1_5/',str(dest)+'/'],stdout=log,stderr=subprocess.STDOUT,check=True)
print('SD15 transfer completed',flush=True)
