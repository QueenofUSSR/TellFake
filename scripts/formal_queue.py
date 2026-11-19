"""Frozen-source, GPU-serial formal training queue. No final evaluation."""
import fcntl,json,os,subprocess,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
QUEUE=ROOT.parent

def now():return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()
def save(value):
 p=QUEUE/'status.tmp';p.write_text(json.dumps(value,indent=2)+'\n');p.replace(QUEUE/'status.json')

def main():
 protocol=json.loads((QUEUE/'protocol.json').read_text())
 state=dict(status='running',started=now(),completed=['A_v3_s42'],current=None,final_predictions_run=False)
 with (QUEUE/'queue.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for run in protocol['pending_runs']:
   state['current']=run['name'];save(state)
   cmd=[sys.executable,'-u','-m','genimage_detector','train','--config',str(ROOT/'configs'/run['config']),'--data-root','/mnt/ssd/data/datasets/GenImage/official_v3','--manifest-dir',str(ROOT/'data/manifests/v3'),'--checkpoint-root','/mnt/ssd/data/ckpt/arin5201','--log-root','/mnt/ssd/data/logs/arin5201','--name',run['name'],'--seed',str(run['seed']),'--device','cuda']
   with (QUEUE/(run['name']+'.log')).open('x') as log:
    code=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
   if code:
    state.update(status='failed',returncode=code,finished=now());save(state);raise RuntimeError('Failed '+run['name'])
   state['completed'].append(run['name']);save(state)
  state.update(status='completed',current=None,finished=now());save(state)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  (QUEUE/'failure.json').write_text(json.dumps(dict(error=repr(e),time=now()),indent=2));raise
