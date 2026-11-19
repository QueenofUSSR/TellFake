"""Pin additional backbone artifacts on SSD, no model training."""
import hashlib,json,os,urllib.request
from pathlib import Path
import torch,torchvision.models as tv
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reproduction/p1';cache=Path('/mnt/ssd/data/cache/arin5201/reproduction/backbones');cache.mkdir(exist_ok=True)
records=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
for name,w in [('resnet18',tv.ResNet18_Weights.IMAGENET1K_V1),('resnet50_v1',tv.ResNet50_Weights.IMAGENET1K_V1),('resnet50_v2',tv.ResNet50_Weights.IMAGENET1K_V2),('swin_t',tv.Swin_T_Weights.IMAGENET1K_V1)]:
 state=w.get_state_dict(progress=False,check_hash=True);path=Path(torch.hub.get_dir())/'checkpoints'/w.url.rsplit('/',1)[-1];records.append(dict(model=name,url=w.url,path=str(path),sha256=sha(path),bytes=path.stat().st_size,version=str(w),license_note='torchvision code BSD; pretrained ImageNet weights retain dataset usage constraints'));del state;print('Verified',name,flush=True)
repo='timm/deit_small_patch16_224.fb_in1k';info=json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+repo+'?blobs=true',timeout=30));rev=info['sha'];file=next(x for x in info['siblings'] if x['rfilename']=='model.safetensors');expected=file['lfs']['sha256'];path=cache/('deit_s_'+rev+'.safetensors')
if not path.exists():
 url='https://huggingface.co/'+repo+'/resolve/'+rev+'/model.safetensors?download=true'
 with urllib.request.urlopen(url,timeout=90) as r,path.with_suffix('.partial').open('wb') as f:
  while b:=r.read(8*1024**2):f.write(b)
 path.with_suffix('.partial').rename(path)
assert sha(path)==expected
records.append(dict(model='deit_s',repo=repo,revision=rev,path=str(path),sha256=expected,bytes=path.stat().st_size,license=(info.get('cardData') or {}).get('license','see model card')))
(cache/'deit_source_metadata.json').write_text(json.dumps(info,indent=2)+'\n');(OUT/'extra_weights.lock.json').write_text(json.dumps(records,indent=2)+'\n');print('All extra backbones pinned',flush=True)
