#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
result_dir="${1:?Provide a NEW output directory for reproducible summary}"
report_path="${2:?Provide a NEW PDF output path}"
if [[ -e "$result_dir" || -e "$report_path" ]]; then
  echo 'Refusing to overwrite previous results' >&2
  exit 1
fi
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR=/mnt/ssd/cache/matplotlib
.venv-local/bin/python scripts/summarize_final.py "$result_dir"
.venv-local/bin/python scripts/plot_final.py "$result_dir"
pdf_python="${PDF_PYTHON:-/home/tom/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"
"$pdf_python" scripts/report_final.py "$result_dir" "$report_path"
