import json,sys,subprocess,hashlib,shutil
from pathlib import Path
from PIL import Image
root=Path('/home/tom/文档/GenImage/midjourney_subset_work');batch=json.loads(sys.stdin.read()); dest=root/'batch'
if dest.exists():raise RuntimeError('Previous batch not cleared')
if shutil.disk_usage(root).free < 5*1024**3:raise RuntimeError('Remote space low')
listing=root/'selected.txt';listing.write_text('\n'.join(f['path'] for f in batch)+'\n')
subprocess.run([str(root/'tools/extracted/usr/lib/7zip/7z'),'x','-y','-bso0','-bsp0','-o'+str(dest),'/home/tom/文档/GenImage/Midjourney/imagenet_midjourney.zip','@'+str(listing)],check=True)
for f in batch:
 p=dest/f['path']
 if p.stat().st_size!=f['size']:raise RuntimeError('Size mismatch')
 with Image.open(p) as im:im.load()
 f['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
(dest/'manifest.json').write_text(json.dumps(batch))
print('REMOTE_VERIFIED',len(batch),flush=True)
