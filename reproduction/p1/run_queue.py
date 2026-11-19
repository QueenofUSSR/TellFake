"""Serial GPU S0 queue; failures retained and completed outputs never overwritten."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'reproduction/p1'
def main():
 cfg=json.loads((P/'protocol.json').read_text());out=Path(cfg['output_root']);results=[]
 with (out/'gpu_queue.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for method in sys.argv[1:] or cfg['methods']:
   if method not in cfg['methods']:raise ValueError(method)
   target=out/method;target.mkdir(parents=True,exist_ok=True)
   if (target/'result.json').exists():results.append([method,'already_passed']);continue
   logfile=target/('run_'+str(time.time_ns())+'.log');print('START',method,flush=True)
   with logfile.open('w') as log:
    proc=subprocess.run([sys.executable,str(P/'run_method.py'),'--method',method],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   status='passed' if proc.returncode==0 else 'failed';results.append([method,status]);print('END',method,status,flush=True)
   (out/'queue_status.json').write_text(json.dumps(results,indent=2)+'\n')
   if proc.returncode:break
if __name__=='__main__':main()
