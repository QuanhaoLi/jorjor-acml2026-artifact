#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/k/k.0.8_strong_teacher_seed_scale_remote}"
DEVICE="${DEVICE:-cuda}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" scripts/build_teacher_strength_seed_scale_pack.py \
  --output-dir "${OUTPUT_DIR}" \
  --seeds 20260611 20260612 20260613 \
  --teacher-iterations 10 \
  --teacher-deal-sample-count 32 \
  --teacher-hand-count 1000 \
  --teacher-evaluation-hand-count 300 \
  --student-epochs 8 \
  --student-batch-size 64 \
  --student-hidden-size 48 \
  --student-num-heads 4 \
  --student-num-layers 4 \
  --probe-epochs 20 \
  --probe-batch-size 128 \
  --bootstrap-count 1000 \
  --minimum-matched-points 20 \
  --locked-layer encoder_layer_2 \
  --device "${DEVICE}"
