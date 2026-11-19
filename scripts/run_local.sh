#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TORCH_HOME=/mnt/ssd/cache/torch
export XDG_CACHE_HOME=/mnt/ssd/cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export GENIMAGE_ROOT=/mnt/ssd/data/datasets/GenImage/official_v3
export GENIMAGE_CHECKPOINT_ROOT=/mnt/ssd/data/ckpt/arin5201
export GENIMAGE_LOG_ROOT=/mnt/ssd/data/logs/arin5201
cd "$project_root"
if [[ "${1:-}" == test ]]; then
  exec "$project_root/.venv-local/bin/python" -m unittest discover -s tests -v
fi
exec "$project_root/.venv-local/bin/python" -u -m genimage_detector "$@"
