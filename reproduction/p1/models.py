"""P1 native method bindings. No final-test loader and no upstream source edits."""
import contextlib,hashlib,importlib.util,io,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageFilter
import torch
from torch import nn
import torchvision.transforms as T
from transformers import CLIPVisionModel,CLIPVisionModelWithProjection,CLIPImageProcessor
ROOT=Path(__file__).resolve().parents[2]
SRC=Path('/mnt/ssd/data/cache/arin5201/reproduction/sources')
WEIGHTS={r['repo']:r['local_path'] for r in json.loads((ROOT/'reproduction/weights.lock.json').read_text())}
MEAN=(.48145466,.4578275,.40821073);STD=(.26862954,.26130258,.27577711)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def native_mixin(method):
 old=sys.argv[:];sys.argv=['native_train','/mnt/ssd/data/cache/arin5201/reproduction',method,'P1','/mnt/ssd/data/logs/arin5201/reproduction/p1_20261114/native_import', '1']
 os.environ['REPRO_CLIP_PATH']=WEIGHTS['openai/clip-vit-large-patch14'];os.environ['CHROMA_FIX']='0'
 try:m=module('p1_mixin',SRC/'ARIN5201-AIGI-Detection/code/exp/train_ft.py')
 finally:sys.argv=old
 # Cache the EXACT native CPU SVD, keyed by input bytes and native implementation.
 original=m.SVDResidualLinear;cache=Path('/mnt/ssd/data/cache/arin5201/reproduction/svd');cache.mkdir(exist_ok=True)
 source_hash=hashlib.sha256((SRC/'ARIN5201-AIGI-Detection/code/exp/train_ft.py').read_bytes()).hexdigest()
 class CachedSVD(original):
  def __init__(self,lin,r=1023):
   key=hashlib.sha256(lin.weight.detach().cpu().numpy().tobytes()+str(r).encode()+source_hash.encode()).hexdigest();p=cache/(key+'.pt')
   if p.exists():
    nn.Module.__init__(self);d=torch.load(p,weights_only=True,map_location='cpu');self.register_buffer('weight_main',d['weight_main'])
    for k in ['U_residual','S_residual','V_residual']:setattr(self,k,nn.Parameter(d[k]))
    self.bias=nn.Parameter(lin.bias.detach().clone(),requires_grad=False) if lin.bias is not None else None
   else:
    super().__init__(lin,r);torch.save({k:v.detach().cpu() for k,v in self.state_dict().items() if k!='bias'},p)
 m.SVDResidualLinear=CachedSVD;return m

class SingleBranch(nn.Module):
 def __init__(self,kind):
  super().__init__();d=module('p1_dual',SRC/'clip_dual_aigi/models.py');self.encoder=CLIPVisionModel.from_pretrained(WEIGHTS['openai/clip-vit-large-patch14'],local_files_only=True);self.encoder.requires_grad_(False);self.kind=kind
  if kind=='adapter_only':d._apply_adapters(self.encoder,64);d._freeze_except(self.encoder,(d.BottleneckAdapter,))
  if kind=='ssf_only':d._apply_ssf(self.encoder);d._freeze_except(self.encoder,(d.SSF,))
  if kind!='frozen_only':self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  self.head=nn.Linear(self.encoder.config.hidden_size,1)
 def train(self,mode=True):
  super().train(mode)
  if self.kind=='frozen_only':self.encoder.eval()
  return self
 def forward(self,x):
  with torch.no_grad() if self.kind=='frozen_only' else contextlib.nullcontext():f=self.encoder(x).pooler_output
  return self.head(f).flatten()

def iapl_args(tta=False):
 sys.path.insert(0,str(SRC/'IAPL'));from main import get_args_parser
 args=get_args_parser().parse_args(['--train_selected_subsets','SD14','--test_selected_subsets','validation','--frozen-manifest-dir',str(ROOT/'reproduction/manifests/v1_20261114'),'--clip-path',str(Path(WEIGHTS['OpenAI CLIP original JIT'])/'ViT-L-14.pt'),'--dataset','GenImage','--gate','True','--condition','True','--smooth','True','--epoch','1','--lr','0.00005'])
 if tta:args.tta=True;args.tta_steps=2;args.ois=True;args.lr=.005
 return args

def build(method):
 if method.startswith('frozen_') and method!='frozen_only':
  variant='large-patch14' if 'L14' in method else 'base-patch16' if 'B16' in method else 'base-patch32'
  p=WEIGHTS['openai/clip-vit-'+variant];m=CLIPVisionModelWithProjection.from_pretrained(p,local_files_only=True);m.requires_grad_(False);return m,dict(kind='features',path=p,lr=None,optimizer=None,amp=False,native=None)
 if method.startswith('dual_'):
  d=module('p1_dual',SRC/'clip_dual_aigi/models.py');m=d.DualCLIPDetector(WEIGHTS['openai/clip-vit-large-patch14'],method=method)
  return m,dict(kind='binary',lr=2e-4,optimizer='AdamW',weight_decay=0.,amp=True,native=d)
 if method in ['frozen_only','adapter_only','ssf_only']:return SingleBranch(method),dict(kind='binary',lr=2e-4,optimizer='AdamW',weight_decay=0.,amp=True,native=None)
 if method=='iapl':
  args=iapl_args();from models import build_model
  m=build_model(args);return m,dict(kind='iapl',lr=5e-5,optimizer='Adam',weight_decay=args.weight_decay,amp=False,native=args)
 base='lora' if method.startswith('lora') else 'effort';native=native_mixin(base);m=native.build(base)
 return m,dict(kind='two_class',lr=1e-4 if base=='lora' else 2e-4,optimizer='Adam',weight_decay=5e-4,amp=True,native=native)

class Transform:
 def __init__(self,method,train,meta):
  self.method=method;self.train=train;self.native=meta.get('native');self.norm=T.Compose([T.ToTensor(),T.Normalize(MEAN,STD)])
  if meta['kind']=='features':
   self.proc=CLIPImageProcessor.from_pretrained(meta['path'],local_files_only=True);self.thomas=module('p1_thomas',SRC/'ARIN5201_CV_Project/scripts/train_baseline.py');self.crop=self.thomas.CropPreprocess(128,self.proc.image_mean,self.proc.image_std)
  elif method=='iapl':
   from utils.dataset import Dataset_Creator_GenImage
   self.tf=Dataset_Creator_GenImage('',32).transforms['train' if train else 'val']
  elif method in ['dual_adapter','dual_ssf','adapter_only','ssf_only','frozen_only']:
   d=module('p1_dual_data',SRC/'clip_dual_aigi/dataset.py');self.tf=d.train_transform() if train else d.eval_transform()
 def __call__(self,img,seed):
  method=self.method
  if method.startswith('frozen_') and method!='frozen_only':
   rng=np.random.default_rng(seed)
   jrng=np.random.default_rng(seed+1000)
   if self.train and method.endswith('_jpeg') and jrng.random()<.5:img=self.thomas.jpeg_aug(img,jrng,55,100)
   if 'crop128' in method:return torch.from_numpy(self.crop(img,rng if self.train else None))
   return self.proc(images=img,return_tensors='pt')['pixel_values'][0]
  if method=='iapl' or method in ['dual_adapter','dual_ssf','adapter_only','ssf_only','frozen_only']:return self.tf(img)
  if self.train and method.endswith('_native'):return self.native.NORM(self.native.common_aug(img))
  if self.train and method=='effort_degradation':
   op=random.Random(seed+2000).randrange(10)
   if 1<=op<=3:
    b=io.BytesIO();img.save(b,'JPEG',quality=[90,70,50][op-1],subsampling=2);b.seek(0);img=Image.open(b).convert('RGB')
   elif 4<=op<=6:img=img.filter(ImageFilter.GaussianBlur([.5,1,2][op-4]))
   elif op>=7:
    w,h=img.size;scale=[.75,.5,.25][op-7];img=img.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.BICUBIC).resize((w,h),Image.Resampling.BICUBIC)
  if self.train and random.random()<.5:img=T.functional.hflip(img)
  return self.native.NORM(self.native.pad_crop(img,rand=self.train))

class Images(torch.utils.data.Dataset):
 def __init__(self,rows,transform,seed=42,epoch=0):self.rows=rows;self.transform=transform;self.seed=seed;self.epoch=epoch
 def __len__(self):return len(self.rows)
 def __getitem__(self,index):
  r=self.rows[index];seed=int.from_bytes(hashlib.sha256(f'{self.seed}|{self.epoch}|{r["original_id"]}'.encode()).digest()[:4],'big')
  # Explicit per-original/epoch randomness: worker scheduling cannot alter replay.
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  with Image.open(r['path']) as im:im.load();img=ImageOps.exif_transpose(im).convert('RGB')
  return self.transform(img,seed),int(r['label']),index

def scores(output,kind):
 if kind=='iapl' and isinstance(output,list):output=output[0]
 return (output[:,1]-output[:,0]).float() if kind=='two_class' else output.flatten().float()
def loss(model,x,y,kind):
 output=model(x)
 if kind=='iapl':
  terms=model.get_criterion(output,y);return sum(terms[k]*model.criterion_weight_dict[k] for k in terms if k in model.criterion_weight_dict)
 return torch.nn.functional.cross_entropy(output.float(),y) if kind=='two_class' else torch.nn.functional.binary_cross_entropy_with_logits(output.flatten().float(),y.float())
