#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/k/k.2_llm_latent_order_remote}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"
LAYER_ID="${LAYER_ID:--1}"
MAX_PAIRS="${MAX_PAIRS:-512}"
BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-1000}"
BACKENDS="${BACKENDS:-oracle_posterior,composition,order_only,surface_noise,hf_causal_lm}"
HF_BATCH_SIZE="${HF_BATCH_SIZE:-16}"

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import subprocess
import sys

missing = [
    package
    for package in ("transformers", "tokenizers", "safetensors", "huggingface_hub")
    if importlib.util.find_spec(package) is None
]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY

"${PYTHON_BIN}" scripts/build_llm_latent_order_report.py \
  --output-dir "${OUTPUT_DIR}" \
  --length 5 \
  --max-pairs "${MAX_PAIRS}" \
  --bootstrap-count "${BOOTSTRAP_COUNT}" \
  --backends "${BACKENDS}" \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --layer-id "${LAYER_ID}" \
  --hf-batch-size "${HF_BATCH_SIZE}" \
  --token-selection final
