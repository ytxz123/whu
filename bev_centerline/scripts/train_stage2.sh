#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m centerline_mm.train.stage2_train_alignment --config configs/stage2_alignment.yaml

