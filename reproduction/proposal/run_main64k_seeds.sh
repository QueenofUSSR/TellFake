#!/usr/bin/env bash
# Sequential supplement to the completed 64k seed-42 core runs.
set -euo pipefail
cd /mnt/ssd/code/arin5201
log_root=/mnt/ssd/data/logs/arin5201/reproduction/proposal_main64k_seeds43_44_20261118
mkdir -p "$log_root"
exec 9>"$log_root/queue.lock"
flock -n 9
for seed in 43 44; do
    printf 'START seed %s %s\n' "$seed" "$(date -Iseconds)"
    bash reproduction/run.sh reproduction/proposal/run_scale.py \
        --config "reproduction/proposal/main64k_seed${seed}.json"
    printf 'COMPLETE seed %s %s\n' "$seed" "$(date -Iseconds)"
done
