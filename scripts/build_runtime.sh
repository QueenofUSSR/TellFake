#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_image="${GENIMAGE_BASE_IMAGE:-assignment2:torch2.5.1-cu124-v1}"
runtime_image="${GENIMAGE_IMAGE:-arin5201:torch2.5.1-cu124-v1}"
docker image inspect "$base_image" --format 'Base image: {{.Id}}'
docker build --build-arg "BASE_IMAGE=$base_image" -t "$runtime_image" "$project_root"
docker image inspect "$runtime_image" --format 'Runtime image: {{.Id}}'
