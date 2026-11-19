"""PEFT delta checkpoint: immutable omitted tensors are verified on every load."""
import hashlib,torch

def tensor_hash(t):
 return hashlib.sha256(str(t.dtype).encode()+str(tuple(t.shape)).encode()+t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def make_delta(full,model):
 trainable={n for n,p in model.named_parameters() if p.requires_grad}
 mutable_buffers={n for n,b in model.named_buffers() if not n.endswith('weight_main')}
 included=trainable|mutable_buffers
 state=full['model'];out=dict(full);out['format']='verified_delta_v1'
 out['model']={n:t for n,t in state.items() if n in included}
 out['base_sha256']={n:tensor_hash(t) for n,t in state.items() if n not in included}
 out['nonpersistent_buffers']={n:b.detach().cpu().clone() for n,b in model.named_buffers() if n not in state}
 return out

def load_delta(model,checkpoint,optimizer=None,scheduler=None,scaler=None):
 if checkpoint['format']!='verified_delta_v1':raise ValueError('Unknown checkpoint format')
 current=model.state_dict();included=set(checkpoint['model']);omitted=set(checkpoint['base_sha256'])
 if included&omitted or included|omitted != set(current):raise ValueError('Architecture/checkpoint key mismatch')
 for n,expected in checkpoint['base_sha256'].items():
  if tensor_hash(current[n])!=expected:raise ValueError('Frozen base mismatch: '+n)
 result=model.load_state_dict(checkpoint['model'],strict=False)
 if set(result.missing_keys)!=omitted or result.unexpected_keys:raise ValueError('Incomplete delta load')
 buffers=dict(model.named_buffers())
 if set(checkpoint['nonpersistent_buffers']) != set(buffers)-set(current):raise ValueError('Nonpersistent buffer mismatch')
 with torch.no_grad():
  for n,b in checkpoint['nonpersistent_buffers'].items():buffers[n].copy_(b)
 if optimizer is not None:optimizer.load_state_dict(checkpoint['optimizer'])
 if scheduler is not None:scheduler.load_state_dict(checkpoint['scheduler'])
 if scaler is not None:scaler.load_state_dict(checkpoint['scaler'])
 return {'next_batch':checkpoint['next_batch'],'sampler':checkpoint['sampler'],'rng':checkpoint['rng']}
