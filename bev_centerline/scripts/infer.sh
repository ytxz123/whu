#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m centerline_mm.infer.generate_json --config configs/infer.yaml "$@"

