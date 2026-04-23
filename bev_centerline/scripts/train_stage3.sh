#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m centerline_mm.train.stage3_train_json_lora --config configs/stage3_json_lora.yaml

