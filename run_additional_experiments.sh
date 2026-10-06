#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$PROJECT_DIR/.ultralytics}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
PROJECT_PYTHON="${OBJECT_DETECTION_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
if [[ ! -x "$PROJECT_PYTHON" ]]; then
    echo "Python environment unavailable: $PROJECT_PYTHON" >&2
    exit 1
fi
exec "$PROJECT_PYTHON" tools/run_additional_experiments.py "$@"
