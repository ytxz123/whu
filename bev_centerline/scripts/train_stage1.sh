#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m centerline_mm.train.stage1_train_seg --config configs/stage1_segmentation.yaml

