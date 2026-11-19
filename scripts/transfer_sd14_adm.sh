#!/bin/bash
set -euo pipefail
cd /mnt/ssd/code/arin5201
for spec in 'SD14 stable_diffusion_v_1_4 sd14' 'ADM ADM adm'; do
 read -r name remote logname <<< "$spec"
 mkdir -p "/mnt/ssd/data/downloads/arin5201/lan_transfer/$name"
 python3 - "$name" <<'PY'
import shutil,sys
minimum={'SD14':320,'ADM':165}[sys.argv[1]]*1024**3
if shutil.disk_usage('/mnt/ssd').free<minimum:raise SystemExit('Insufficient space; stopped before transfer')
PY
 echo "Starting $name transfer"
 rsync -rt --partial --checksum --info=progress2 -e 'ssh -S /tmp/arin5201-transfer-ssh3' "tom@192.168.60.1:/home/tom/文档/GenImage/$remote/" "/mnt/ssd/data/downloads/arin5201/lan_transfer/$name/" > "/mnt/ssd/data/logs/arin5201/downloads_v3/lan_${logname}_transfer.log" 2>&1
 echo "Starting $name verification and extraction"
 uv run --with pillow python "scripts/finish_lan_${logname}.py" > "/mnt/ssd/data/logs/arin5201/downloads_v3/lan_${logname}_pipeline.log" 2>&1
 echo "Completed $name"
done
