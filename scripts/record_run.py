"""Record a detached command's real exit status; never treat failure as zero."""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def now():return datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()

def main():
    name,command=sys.argv[1],sys.argv[2:]
    if Path(name).name!=name or not command:raise ValueError('Need unique name and command')
    folder=Path('/mnt/ssd/data/logs/arin5201/launches')/name
    folder.mkdir(parents=True,exist_ok=False)
    state=dict(name=name,command=command,started=now(),wrapper_pid=os.getpid(),status='starting')
    def save():
        temp=folder/'status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(folder/'status.json')
    save()
    try:
        with (folder/'console.log').open('x') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            state.update(pid=process.pid,status='running');save()
            result=process.wait()
        state.update(status='completed' if result==0 else 'failed',returncode=result,finished=now());save()
        return result
    except BaseException as e:
        state.update(status='failed',error=repr(e),finished=now());save();raise

if __name__=='__main__':sys.exit(main())
