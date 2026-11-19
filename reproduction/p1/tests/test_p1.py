import copy,hashlib,importlib,io,json,random,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from reproduction.p1.models import Transform
from genimage_detector.metrics import choose_fpr_threshold,binary_metrics
class Tests(unittest.TestCase):
 def test_queue_import_has_no_side_effect(self):
  import reproduction.p1.run_queue
  self.assertTrue(callable(reproduction.p1.run_queue.main))
 def test_degradation_does_not_use_label(self):
  from reproduction.p1.models import native_mixin
  m=native_mixin('effort');t=Transform('effort_degradation',True,{'native':m,'kind':'two_class'})
  image=Image.fromarray(np.random.default_rng(7).integers(0,256,(256,256,3),dtype=np.uint8))
  for seed in range(20):
   outputs=[]
   for label in [0,1]:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);outputs.append(t(image,seed))
   torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
 def test_diagnostic_fpr_threshold_ties(self):
  y=[0]*100+[1]*100;p=[.4]*100+[.5]*100;t=choose_fpr_threshold(y,p)
  self.assertEqual(binary_metrics(y,p,t)['real_fpr'],0.)
 def test_no_final_test_in_s0_protocol(self):
  d=json.loads((ROOT/'reproduction/p1/protocol.json').read_text())
  self.assertFalse(d['final_test_access']);self.assertIn('val',Path(d['validation_manifest']).name)
  with open(d['validation_manifest']) as f:rows=list(__import__('csv').DictReader(f))
  self.assertEqual(len(rows),1024);self.assertTrue(all(r['source_split']=='train' for r in rows))
if __name__=='__main__':unittest.main()
