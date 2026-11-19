import copy,unittest
import torch
from genimage_detector.engine import finite_amp_step

class AmpReplayTests(unittest.TestCase):
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
 def test_overflow_replays_batch_and_restores_bn_buffers(self):
  torch.manual_seed(42)
  model=torch.nn.Sequential(torch.nn.BatchNorm1d(4),torch.nn.Linear(4,1)).cuda()
  reference=copy.deepcopy(model);x=torch.randn(8,4,device='cuda')
  optim=torch.optim.SGD(model.parameters(),lr=.01);refoptim=torch.optim.SGD(reference.parameters(),lr=.01)
  calls=[0]
  def hook(g):
   calls[0]+=1
   return g*float('inf') if calls[0]==1 else g
  handle=model[1].weight.register_hook(hook)
  def closure():
   loss=model(x).square().mean();return loss,loss,loss*0
  _,norm,retries=finite_amp_step(model,optim,torch.amp.GradScaler('cuda'),closure)
  reference(x).square().mean().backward();refoptim.step()
  self.assertEqual(retries,1)
  for a,b in zip(model.state_dict().values(),reference.state_dict().values()):torch.testing.assert_close(a,b)
  handle.remove()
