"""Pin remote revisions, verify LFS SHA256, download only required model files."""
import concurrent.futures,hashlib,json,os,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DEST=Path('/mnt/ssd/data/cache/arin5201/reproduction/weights');DEST.mkdir(parents=True,exist_ok=True)
MODELS=[('openai/clip-vit-large-patch14','32bd64288804d66eefd0ccbe215aa642df71cc41'),('openai/clip-vit-base-patch16','57c216476eefef5ab752ec549e440a49ae4ae5f3'),('openai/clip-vit-base-patch32','3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'),('sd2-community/stable-diffusion-2-1','bb2154823665391b4fb29b0b9cf82a198964ee05')]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def get(url):
 for attempt in range(5):
  try:
   with urllib.request.urlopen(url,timeout=60) as f:return f.read()
  except Exception:
   if attempt==4:raise
   time.sleep(1)
def fetch(job):
 url,p,expected=job;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and (not expected or sha(p)==expected):return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,url=url,upstream_lfs_sha256=expected)
 for attempt in range(4):
  try:
   with urllib.request.urlopen(url+('?download=true&nonce='+str(time.time_ns()) if '?' not in url else ''),timeout=90) as r,p.with_suffix(p.suffix+'.partial').open('wb') as out:
    while b:=r.read(8*1024**2):out.write(b)
   tmp=p.with_suffix(p.suffix+'.partial');h=sha(tmp)
   if expected and h!=expected:raise RuntimeError('Upstream hash mismatch '+str(p))
   tmp.rename(p);print('Verified',p.name,p.stat().st_size,flush=True)
   return dict(path=str(p),sha256=h,bytes=p.stat().st_size,url=url,upstream_lfs_sha256=expected)
  except Exception as e:
   print('Retry',p.name,attempt,type(e).__name__,str(e)[:150],flush=True)
   if attempt==3:raise
records=[];jobs=[]
for repo,rev in MODELS:
 folder=DEST/(repo.replace('/','--')+'--'+rev[:12]);folder.mkdir(exist_ok=True)
 meta=folder/'source_metadata.json'
 info=json.loads(meta.read_text()) if meta.exists() else json.loads(get('https://huggingface.co/api/models/'+repo+'/revision/'+rev+'?blobs=true'))
 assert info['sha']==rev
 meta.write_text(json.dumps(info,indent=2))
 print('Pinned',repo,rev,flush=True)
 names={s['rfilename']:s for s in info['siblings']}
 if repo.startswith('sd2-'):wanted=['vae/config.json','vae/diffusion_pytorch_model.safetensors','README.md']
 else:wanted=['config.json','preprocessor_config.json','model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin','README.md']
 rec=dict(repo=repo,revision=rev,local_path=str(folder),license=(info.get('cardData') or {}).get('license','MIT (OpenAI CLIP upstream; HF card does not declare)'),files=[])
 for name in wanted:
  if name not in names:raise RuntimeError('Missing asset '+repo+'/'+name)
  expected=(names[name].get('lfs') or {}).get('sha256')
  jobs.append((rec,('https://huggingface.co/'+repo+'/resolve/'+rev+'/'+name,folder/name,expected)))
 records.append(rec)
# IAPL's bundled OpenAI CLIP loader requires the original JIT format.
import ast
clip_rev=json.loads(get('https://api.github.com/repos/openai/CLIP/commits/main'))['sha']
import base64
source=base64.b64decode(json.loads(get('https://api.github.com/repos/openai/CLIP/contents/clip/clip.py?ref='+clip_rev))['content']).decode();(DEST/'openai-clip-source.py').write_text(source);(DEST/'openai-clip-LICENSE').write_bytes(base64.b64decode(json.loads(get('https://api.github.com/repos/openai/CLIP/contents/LICENSE?ref='+clip_rev))['content']));tree=ast.parse(source)
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_MODELS' for t in node.targets):urls=ast.literal_eval(node.value)
url=urls['ViT-L/14'];expected=url.split('/')[-2]
rec=dict(repo='OpenAI CLIP original JIT',revision=expected,local_path=str(DEST/'openai-jit'),license='MIT',files=[]);records.append(rec);jobs.append((rec,(url,DEST/'openai-jit/ViT-L-14.pt',expected)))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 for (rec,_),result in zip(jobs,ex.map(fetch,[j for _,j in jobs])):rec['files'].append(result)
(ROOT/'reproduction/weights.lock.json').write_text(json.dumps(records,indent=2)+'\n')
print('All weight files verified',flush=True)
