import csv,hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from reproduction.protocol import assert_disjoint,derived_row,read_manifest

class ProtocolTests(unittest.TestCase):
 def test_reencoded_pixel_leakage(self):
  a=dict(original_id='a',group_id='a',sha256='a',pixel_sha256='same')
  b=dict(original_id='b',group_id='b',sha256='b',pixel_sha256='same')
  with self.assertRaisesRegex(ValueError,'pixel_sha256'):assert_disjoint([a],[b])
 def test_original_identity_leakage(self):
  a=dict(original_id='a',group_id='a',sha256='a',pixel_sha256='a')
  b=dict(original_id='a',group_id='b',sha256='b',pixel_sha256='b')
  with self.assertRaisesRegex(ValueError,'original_id'):assert_disjoint([a],[b])
 def test_derivative_cannot_change_partition(self):
  p=dict(original_id='a',group_id='g',split='test',source_split='val',sha256='s')
  self.assertEqual(derived_row(p,'jpeg')['split'],'test')
  with self.assertRaises(ValueError):derived_row(p,'jpeg',split='train')
 def test_manifest_tampering(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'train.csv';p.write_text('path,label\n')
   (p.parent/'protocol.json').write_text(json.dumps({'manifest_sha256':{'train':hashlib.sha256(p.read_bytes()).hexdigest()}}))
   p.write_text('path,label\ncorrupted,1\n')
   with self.assertRaisesRegex(ValueError,'modified'):read_manifest(p)
 def test_missing_source_never_falls_back(self):
  import sys
  src=Path('/mnt/ssd/data/cache/arin5201/reproduction/sources/clip_dual_aigi/dataset.py')
  spec=importlib.util.spec_from_file_location('dual_dataset',src);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  with tempfile.TemporaryDirectory() as tmp:
   (Path(tmp)/'SD14/train/ai').mkdir(parents=True)
   with self.assertRaises(FileNotFoundError):m.build_rows(tmp,'SD14','val')
 def test_partial_accumulation_window(self):
  import torch
  x=torch.arange(1.,6.);y=2*x
  a=torch.nn.Linear(1,1,bias=False);b=torch.nn.Linear(1,1,bias=False);b.load_state_dict(a.state_dict())
  oa=torch.optim.SGD(a.parameters(),lr=.001);ob=torch.optim.SGD(b.parameters(),lr=.001)
  for start in range(0,5,3):
   oa.zero_grad();loss=((a(x[start:start+3,None]).flatten()-y[start:start+3])**2).mean();loss.backward();oa.step()
  ob.zero_grad()
  for step in range(1,6):
   window=min(3,5-((step-1)//3)*3)
   loss=((b(x[step-1:step,None]).flatten()-y[step-1:step])**2).mean()/window;loss.backward()
   if step%3==0 or step==5:ob.step();ob.zero_grad()
  torch.testing.assert_close(a.weight,b.weight)
if __name__=='__main__':unittest.main()
