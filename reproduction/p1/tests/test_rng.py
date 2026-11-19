import json,random,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
from reproduction.p1.models import Transform,WEIGHTS,native_mixin
class RNGIsolationTests(unittest.TestCase):
 def test_jpeg_does_not_advance_crop_rng(self):
  meta=dict(kind='features',path=WEIGHTS['openai/clip-vit-base-patch16'])
  base=Transform('frozen_B16_crop128',True,meta);aug=Transform('frozen_B16_crop128_jpeg',True,meta)
  def identity_jpeg(im,rng,*args):rng.integers(55,101);rng.choice([0,2]);return im.copy()
  aug.thomas.jpeg_aug=identity_jpeg
  image=Image.fromarray(np.random.default_rng(1).integers(0,256,(256,320,3),dtype=np.uint8))
  for seed in range(20):torch.testing.assert_close(base(image,seed),aug(image,seed),rtol=0,atol=0)
 def test_degradation_does_not_advance_crop_flip_rng(self):
  native=native_mixin('effort');meta=dict(kind='two_class',native=native)
  base=Transform('effort',True,meta);aug=Transform('effort_degradation',True,meta);image=Image.new('RGB',(256,320),(120,120,120))
  for seed in range(20):
   random.seed(seed);base(image,seed);state_a=random.getstate()
   random.seed(seed);aug(image,seed);state_b=random.getstate()
   self.assertEqual(state_a,state_b)
if __name__=='__main__':unittest.main()
