import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from reproduction.p1.checkpoint import make_delta,load_delta
class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__();self.base=torch.nn.Parameter(torch.tensor(2.),requires_grad=False);self.head=torch.nn.Parameter(torch.tensor([3.]));self.register_buffer('running',torch.tensor([4.]));self.register_buffer('position_ids',torch.tensor([0,1,2]),persistent=False)
 def forward(self,x):return (self.base*x+self.head)*self.running+self.position_ids[1]
class CheckpointTests(unittest.TestCase):
 def payload(self,m):return dict(model={n:t.clone() for n,t in m.state_dict().items()},optimizer={},scheduler={},scaler={},rng={},next_batch=3,sampler={'cursor':3})
 def test_nonpersistent_buffers_restore(self):
  m=Tiny();before=m(torch.ones(1)).detach().clone();d=make_delta(self.payload(m),m)
  with torch.no_grad():m.head.zero_();m.running.zero_();m.position_ids.zero_()
  load_delta(m,d);torch.testing.assert_close(before,m(torch.ones(1)),rtol=0,atol=0)
 def test_wrong_frozen_base_rejected(self):
  m=Tiny();d=make_delta(self.payload(m),m)
  with torch.no_grad():m.base.add_(1)
  with self.assertRaisesRegex(ValueError,'Frozen base mismatch'):load_delta(m,d)
if __name__=='__main__':unittest.main()
