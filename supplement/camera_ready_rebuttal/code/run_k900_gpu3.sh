#!/usr/bin/env bash

set -Eeuo pipefail

GPU_INDEX="${GPU_INDEX:-3}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/home/a/jorjor_runs/k800_five_seed_20260831}"
PRODUCT_DIR="${PRODUCT_DIR:-$REPO_ROOT/outputs/k/k.9.0_exact_posterior}"
STATUS_DIR="${STATUS_DIR:-$REPO_ROOT/outputs/k/k.9.0_remote_status}"
RUN_LOG="${RUN_LOG:-$STATUS_DIR/commands.log}"
LOCK_FILE="${LOCK_FILE:-/tmp/jorjor-gpu3.lock}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-512}"
GPU_UTIL_LIMIT="${GPU_UTIL_LIMIT:-10}"

if [ "$GPU_INDEX" != "3" ]; then
  printf 'k.9.0 is authorized only for physical GPU 3; got %s\n' "$GPU_INDEX" >&2
  exit 2
fi
if [ -z "$SOURCE_REVISION" ]; then
  printf 'SOURCE_REVISION must be the full deployed git revision\n' >&2
  exit 2
fi
if [ ! -x "$PYTHON_BIN" ] || [ ! -d "$ARCHIVE_ROOT" ]; then
  printf 'required Python environment or frozen archive is missing\n' >&2
  exit 2
fi
command -v nvidia-smi >/dev/null 2>&1
command -v flock >/dev/null 2>&1

mkdir -p "$STATUS_DIR"
exec > >(tee -a "$RUN_LOG") 2>&1
exec 9>"$LOCK_FILE"
flock 9
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

log() {
  printf '[k.9.0] %s at %s\n' "$*" "$(date -Iseconds)"
}

on_error() {
  local exit_code=$?
  printf '%s\n' "failed exit_code=$exit_code at $(date -Iseconds)" >"$STATUS_DIR/status.failed"
  log "failed exit_code=$exit_code"
  exit "$exit_code"
}
trap on_error ERR

wait_for_gpu3() {
  while true; do
    local used util pids
    IFS=',' read -r used util < <(
      nvidia-smi --id=3 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
        awk -F, 'NR == 1 {gsub(/ /, "", $1); gsub(/ /, "", $2); print $1 "," $2}'
    )
    pids=$(
      nvidia-smi --id=3 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
        awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/ /, "", $0); print}' |
        paste -sd, -
    )
    if [ -z "$pids" ] && [ "${used:-999999}" -lt "$GPU_MEMORY_LIMIT_MIB" ] && [ "${util:-999999}" -lt "$GPU_UTIL_LIMIT" ]; then
      log "physical GPU3 ready used=${used}MiB util=${util}%"
      return 0
    fi
    log "waiting physical GPU3 used=${used:-unknown}MiB util=${util:-unknown}% pids=${pids:-none}"
    sleep 60
  done
}

run_step() {
  log "start $*"
  "$@"
  log "done $*"
}

main() {
  cd "$REPO_ROOT"
  if [ "$(git rev-parse HEAD)" != "$SOURCE_REVISION" ]; then
    printf 'source revision mismatch\n' >&2
    exit 2
  fi
  wait_for_gpu3
  "$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one visible CUDA device")
print(f"visible_cuda_devices=1 logical_device_0={torch.cuda.get_device_name(0)}")
PY
  rm -f "$STATUS_DIR/status.failed" "$STATUS_DIR/status.complete"
  run_step "$PYTHON_BIN" "$REPO_ROOT/scripts/build_k9_exact_posterior_readouts.py" \
    --archive-root "$ARCHIVE_ROOT" \
    --output-dir "$PRODUCT_DIR" \
    --source-revision "$SOURCE_REVISION" \
    --seed-names seed_k seed_l seed_m seed_n seed_o \
    --device cuda \
    --extraction-batch-size 512 \
    --readout-batch-size 512 \
    --epochs 30 \
    --patience 5 \
    --learning-rate 0.01 \
    --residual-learning-rate 0.002 \
    --weight-decay 0.0001 \
    --bootstrap-count 10000 \
    --bootstrap-seed 20260901 \
    --confidence-level 0.95 \
    --max-pairs-per-bucket 64
  run_step "$PYTHON_BIN" "$REPO_ROOT/scripts/build_k900_environment_manifest.py" \
    --repo-root "$REPO_ROOT" \
    --product-dir "$PRODUCT_DIR" \
    --output "$STATUS_DIR/environment_manifest.json" \
    --source-revision "$SOURCE_REVISION" \
    --physical-gpu-index 3
  printf '%s\n' "complete at $(date -Iseconds)" >"$STATUS_DIR/status.complete"
  log "five-run exact posterior product complete"
}

main "$@"
