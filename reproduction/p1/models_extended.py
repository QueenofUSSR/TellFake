"""Additional proposal/NPR panel, reusing native model and augmentation definitions."""
import json,torch
from pathlib import Path
from reproduction.p1.models import *
from reproduction.p1 import models as core

def build(method):
 native=core.native_mixin(method)
 if method=='deit_s':
  def imagenet_net(m):
   import timm
   from safetensors.torch import load_file
   path=next(r['path'] for r in json.loads((ROOT/'reproduction/p1/extra_weights.lock.json').read_text()) if r['model']=='deit_s')
   net=timm.create_model('deit_small_patch16_224.fb_in1k',pretrained=False,num_classes=1000);net.load_state_dict(load_file(path),strict=True);net.reset_classifier(2);return net
  native.imagenet_net=imagenet_net
 model=native.build(method)
 return model,dict(kind='two_class',lr=1e-4 if method in ['resnet50','deit_s','swin_t','cnnspot'] else 2e-4,optimizer='Adam',weight_decay=5e-4,amp=True,native=native)
class Transform(core.Transform):
 def __call__(self,img,seed):
  return self.native.NORM(self.native.common_aug(img) if self.train else self.native.pad_crop(img,rand=False))
def loss(model,x,y,kind):
 if hasattr(model,'head_sem') and getattr(model,'p1_method','') in ['effort_npr','effort_rgb']:
  out,sem,art=model(x,return_all=True)
  return torch.nn.functional.cross_entropy(out.float(),y)+.5*(torch.nn.functional.cross_entropy(sem.float(),y)+torch.nn.functional.cross_entropy(art.float(),y))
 return core.loss(model,x,y,kind)
_build=build
def build(method):
 model,meta=_build(method);model.p1_method=method;return model,meta
