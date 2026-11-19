"""Synthetic layer compatibility tests, not method S0 or reported experiments."""
import ast,importlib.util,math,sys,unittest
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from transformers import CLIPVisionConfig,CLIPVisionModel
ROOT=Path('/mnt/ssd/data/cache/arin5201/reproduction/sources')
class CompatibilityTests(unittest.TestCase):
 def test_dual_layers_save_load_and_gradients(self):
  spec=importlib.util.spec_from_file_location('dual_models',ROOT/'clip_dual_aigi/models.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  cfg=CLIPVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,image_size=28,patch_size=14)
  for method in ['adapter','ssf']:
   model=CLIPVisionModel(cfg)
   for p in model.parameters():p.requires_grad_(False)
   if method=='adapter':m._apply_adapters(model,4)
   else:m._apply_ssf(model)
   x=torch.randn(2,3,28,28);model.eval();y=model(x).pooler_output
   self.assertTrue(torch.isfinite(y).all())
   model.zero_grad();y.square().mean().backward()
   names=[n for n,p in model.named_parameters() if p.requires_grad]
   self.assertTrue(names);self.assertTrue(all(('adapter_' in n if method=='adapter' else 'ssf_' in n) for n in names))
   self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
   other=CLIPVisionModel(cfg)
   if method=='adapter':m._apply_adapters(other,4)
   else:m._apply_ssf(other)
   other.load_state_dict(model.state_dict());other.eval();torch.testing.assert_close(y,other(x).pooler_output)
 def test_effort_and_lora_initial_equivalence(self):
  tree=ast.parse((ROOT/'ARIN5201-AIGI-Detection/code/exp/train_ft.py').read_text())
  nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['SVDResidualLinear','LoRALinear']]
  space={'torch':torch,'nn':nn,'F':F,'math':math};exec(compile(ast.Module(body=nodes,type_ignores=[]),'upstream_class_extract','exec'),space)
  for name,rank in [('SVDResidualLinear',7),('LoRALinear',2)]:
   base=nn.Linear(8,8);x=torch.randn(2,8);wrapped=space[name](base,r=rank)
   torch.testing.assert_close(wrapped(x),base(x),rtol=1e-4,atol=1e-5)
   wrapped(x).square().mean().backward()
   self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in wrapped.parameters() if p.requires_grad))
if __name__=='__main__':unittest.main()
