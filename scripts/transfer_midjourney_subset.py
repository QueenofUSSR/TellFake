import subprocess,json,hashlib,shutil
from pathlib import Path
base=Path('/mnt/ssd/data/logs/arin5201/midjourney_subset');selection=json.loads((base/'selection.json').read_text())
out=Path('/mnt/ssd/data/datasets/GenImage/official_v3/Midjourney_subset');out.mkdir(parents=True,exist_ok=True)
ssh=['ssh','-S','/tmp/arin5201-transfer-ssh2','tom@192.168.60.1']
remote='/home/tom/文档/GenImage/midjourney_subset_work/'
subprocess.run(['scp','-o','ControlPath=/tmp/arin5201-transfer-ssh2','scripts/midjourney_remote_batch.py','tom@192.168.60.1:'+remote+'run_batch.py'],check=True)
allrows=[]
for idx,batch in enumerate(selection['batches']):
 print('BATCH',idx+1,'of',len(selection['batches']),'extracting',flush=True)
 subprocess.run(ssh+['python3 '+remote+'run_batch.py'],input=json.dumps(batch),text=True,check=True)
 stage=out/('batch_%02d.staging'%idx)
 subprocess.run(['rsync','-rt','--partial','--info=progress2','-e','ssh -S /tmp/arin5201-transfer-ssh2','tom@192.168.60.1:'+remote+'batch/',str(stage)+'/'],check=True)
 rows=json.loads((stage/'manifest.json').read_text())
 for f in rows:
  p=stage/f['path']
  if p.stat().st_size!=f['size'] or hashlib.sha256(p.read_bytes()).hexdigest()!=f['sha256']:raise RuntimeError('Local integrity failure')
 (base/('batch_%02d_verified.json'%idx)).write_text(json.dumps(rows))
 stage.rename(out/('batch_%02d'%idx));allrows.extend(rows)
 # Remove only this job's temporary extracted batch after local verification.
 subprocess.run(ssh+['python3 -c '+__import__('shlex').quote("import shutil; shutil.rmtree('/home/tom/文档/GenImage/midjourney_subset_work/batch')")],check=True)
 print('BATCH_VERIFIED',idx+1,'total_images',len(allrows),flush=True)
result={'status':'complete','images':len(allrows),'bytes':sum(f['size'] for f in allrows),'destination':str(out),'remote_archives_preserved':True}
(base/'complete.json').write_text(json.dumps(result,indent=2));print('COMPLETE',result,flush=True)
