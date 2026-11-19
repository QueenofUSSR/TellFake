"""Verify locked import/API surfaces and load pinned weights without training."""
import ast,gc,hashlib,importlib,json,os,platform,subprocess,sys
from pathlib import Path
import torch
from transformers import CLIPVisionModel,CLIPVisionModelWithProjection,CLIPImageProcessor
from diffusers import AutoencoderKL
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reproduction'
result={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'packages':{}}
for name in ['torchvision','transformers','diffusers','timm','sklearn','pandas','PIL','cv2','pywt','pytorch_wavelets']:
 mod=importlib.import_module(name);result['packages'][name]=getattr(mod,'__version__','import_verified')
# Native entry points import or parse in their own isolated module namespace.
base=Path('/mnt/ssd/data/cache/arin5201/reproduction/sources');env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
checks={}
for name,code in [('IAPL','import main'),('clip_dual_aigi','import models, dataset, train'),('ARIN5201_CV_Project',"import runpy; runpy.run_path('scripts/train_baseline.py',run_name='repro_import')")]:
 out=subprocess.run([sys.executable,'-c',code],cwd=base/name,env=env,capture_output=True,text=True)
 checks[name]={'returncode':out.returncode,'stderr':out.stderr[-3000:]}
 if out.returncode:raise RuntimeError(checks[name])
result['native_imports']=checks
weights=json.loads((R/'weights.lock.json').read_text());loads=[]
for record in weights:
 path=record['local_path'];name=record['repo']
 if name.startswith('openai/'):
  m=CLIPVisionModelWithProjection.from_pretrained(path,local_files_only=True)
  processor=CLIPImageProcessor.from_pretrained(path,local_files_only=True)
  loads.append(dict(repo=name,revision=record['revision'],parameters=sum(p.numel() for p in m.parameters()),hidden_size=m.config.hidden_size,patch_size=m.config.patch_size,processor=processor.to_dict()))
  del m,processor;gc.collect()
 elif name.startswith('sd2-'):
  m=AutoencoderKL.from_pretrained(path,subfolder='vae',local_files_only=True);loads.append(dict(repo=name,revision=record['revision'],parameters=sum(p.numel() for p in m.parameters())));del m;gc.collect()
 else:
  m=torch.jit.load(str(Path(path)/'ViT-L-14.pt'),map_location='cpu');loads.append(dict(repo=name,revision=record['revision'],parameters=sum(p.numel() for p in m.parameters())));del m;gc.collect()
result['weight_loads']=loads;result['status']='passed';result['scope']='Dependency/import/API and actual checkpoint deserialization only; no method training or S0 throughput claim.'
(R/'environment_validation.json').write_text(json.dumps(result,indent=2)+'\n');print('Environment and',len(loads),'weight loads passed')
