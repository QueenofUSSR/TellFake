#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/mnt/ssd/data/cache/arin5201/reproduction
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export TORCH_HOME=/mnt/ssd/cache/torch
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin=/mnt/ssd/data/cache/arin5201/reproduction/env-core/bin/python
cd "$project_root"
case "${1:-}" in
  test) exec "$python_bin" -m unittest discover -s reproduction/tests -v ;;
  validate-data) exec "$python_bin" reproduction/scripts/validate_data.py ;;
  *) exec "$python_bin" "$@" ;;
esac
