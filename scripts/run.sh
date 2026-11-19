#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_dir="${GENIMAGE_DATASET_DIR:-/mnt/ssd/data/datasets/GenImage}"
checkpoint_dir="${GENIMAGE_CHECKPOINT_DIR:-/mnt/ssd/data/ckpt/arin5201}"
log_dir="${GENIMAGE_LOG_DIR:-/mnt/ssd/data/logs/arin5201}"
torch_cache="/mnt/ssd/cache/torch"
runtime_image="${GENIMAGE_IMAGE:-arin5201:torch2.5.1-cu124-v1}"
runtime_id="$(docker image inspect "$runtime_image" --format '{{.Id}}')"
mkdir -p "$checkpoint_dir" "$log_dir" "$torch_cache"
command=(python -m genimage_detector "$@")
if [[ "${1:-}" == "test" ]]; then
  command=(python -m unittest discover -s tests -v)
fi
exec docker run --rm --gpus all --ipc=host --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$project_root,dst=/workspace" \
  --mount "type=bind,src=$dataset_dir,dst=/data/genimage,readonly" \
  --mount "type=bind,src=$checkpoint_dir,dst=/ckpt" \
  --mount "type=bind,src=$log_dir,dst=/logs" \
  --mount "type=bind,src=$torch_cache,dst=/cache/torch" \
  -e GENIMAGE_ROOT=/data/genimage/standard_v1 \
  -e GENIMAGE_CHECKPOINT_ROOT=/ckpt -e GENIMAGE_LOG_ROOT=/logs \
  -e "GENIMAGE_RUNTIME_IMAGE=$runtime_image" -e "GENIMAGE_RUNTIME_ID=$runtime_id" \
  -e TORCH_HOME=/cache/torch -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -w /workspace "$runtime_image" "${command[@]}"
