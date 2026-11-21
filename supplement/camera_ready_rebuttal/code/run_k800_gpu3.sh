#!/usr/bin/env bash

set -Eeuo pipefail

GPU_INDEX="${GPU_INDEX:-3}"
SOURCE_REVISION="${SOURCE_REVISION:-2dfcb42dd7abc9e788666b860c7f4ed0005918c2}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
STATUS_DIR="${STATUS_DIR:-$REPO_ROOT/outputs/k/k.8.0_remote_status}"
RUN_LOG="${RUN_LOG:-$STATUS_DIR/commands.log}"
LOCK_FILE="${LOCK_FILE:-/tmp/jorjor-gpu3.lock}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-512}"
GPU_UTIL_LIMIT="${GPU_UTIL_LIMIT:-10}"

if [ "$GPU_INDEX" != "3" ]; then
  printf 'k.8.0 is authorized only for physical GPU 3; got GPU_INDEX=%s\n' "$GPU_INDEX" >&2
  exit 2
fi
if [ ! -x "$PYTHON_BIN" ]; then
  printf 'Python environment missing or not executable: %s\n' "$PYTHON_BIN" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia-smi is required\n' >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  printf 'flock is required for the GPU3 product lock\n' >&2
  exit 2
fi

mkdir -p "$STATUS_DIR"
exec > >(tee -a "$RUN_LOG") 2>&1
exec 9>"$LOCK_FILE"
flock 9

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

log() {
  printf '[k.8.0] %s at %s\n' "$*" "$(date -Iseconds)"
}

on_error() {
  local exit_code=$?
  printf '%s\n' "failed exit_code=$exit_code at $(date -Iseconds)" >"$STATUS_DIR/status.failed"
  log "failed exit_code=$exit_code"
  exit "$exit_code"
}
trap on_error ERR

verify_source() {
  if [ ! -d "$REPO_ROOT/.git" ] && [ ! -f "$REPO_ROOT/.git" ]; then
    log "source bundle has no git metadata; SOURCE_REVISION=$SOURCE_REVISION"
    return 0
  fi
  local actual_revision
  actual_revision=$(git -C "$REPO_ROOT" rev-parse HEAD)
  if [ "$actual_revision" != "$SOURCE_REVISION" ]; then
    printf 'source revision mismatch: %s != %s\n' "$actual_revision" "$SOURCE_REVISION" >&2
    return 1
  fi
}

verify_cuda_mask() {
  "$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}")
print(f"visible_cuda_devices=1 logical_device_0={torch.cuda.get_device_name(0)}")
PY
}

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

validate_train() {
  local seed="$1"
  local output_dir="$REPO_ROOT/outputs/limit_ar_action_value_only_${seed}_50k_4layer"
  "$PYTHON_BIN" - "$output_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
required = (root / "metrics.json", root / "model.pt", root / "train_log.jsonl")
for path in required:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty training artifact: {path}")
metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
if metrics.get("device") != "cuda":
    raise SystemExit(f"training did not use CUDA: {metrics.get('device')}")
if metrics.get("opponent_hand_supervised") is not False:
    raise SystemExit("training artifact is not no-range supervised")
PY
}

validate_probe() {
  local seed="$1"
  local suffix="$2"
  local output_dir="$REPO_ROOT/outputs/probes/limit_ar_action_value_only_${seed}_4layer_range_${suffix}_50k"
  "$PYTHON_BIN" - "$output_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
required = (root / "probe_metrics.json", root / "probe_table.jsonl", root / "probe_correctness.jsonl")
for path in required:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty probe artifact: {path}")
metrics = json.loads((root / "probe_metrics.json").read_text(encoding="utf-8"))
if metrics.get("device") != "cuda":
    raise SystemExit(f"probe did not use CUDA: {metrics.get('device')}")
if metrics.get("target") != "opponent_hand":
    raise SystemExit(f"unexpected probe target: {metrics.get('target')}")
PY
}

run_train_if_needed() {
  local seed="$1"
  local output_dir="$REPO_ROOT/outputs/limit_ar_action_value_only_${seed}_50k_4layer"
  local present=0
  for filename in metrics.json model.pt train_log.jsonl; do
    [ -e "$output_dir/$filename" ] && present=$((present + 1))
  done
  if [ "$present" -eq 3 ]; then
    validate_train "$seed"
    log "skip complete training seed=$seed"
    return 0
  fi
  if [ "$present" -ne 0 ]; then
    printf 'partial training artifacts require manual recovery: %s\n' "$output_dir" >&2
    return 1
  fi
  run_step "$PYTHON_BIN" -m jorjor train-limit-ar-smoke \
    --config "$REPO_ROOT/configs/train/limit_ar_action_value_only_${seed}_50k_4layer.json"
  validate_train "$seed"
}

run_probe_if_needed() {
  local seed="$1"
  local suffix="$2"
  local output_dir="$REPO_ROOT/outputs/probes/limit_ar_action_value_only_${seed}_4layer_range_${suffix}_50k"
  local present=0
  for filename in probe_metrics.json probe_table.jsonl probe_correctness.jsonl; do
    [ -e "$output_dir/$filename" ] && present=$((present + 1))
  done
  if [ "$present" -eq 3 ]; then
    validate_probe "$seed" "$suffix"
    log "skip complete probe seed=$seed suffix=$suffix"
    return 0
  fi
  if [ "$present" -ne 0 ]; then
    printf 'partial probe artifacts require manual recovery: %s\n' "$output_dir" >&2
    return 1
  fi
  run_step "$PYTHON_BIN" -m jorjor probe-limit-ar \
    --config "$REPO_ROOT/configs/probe/limit_ar_action_value_only_${seed}_4layer_range_${suffix}_50k.json"
  validate_probe "$seed" "$suffix"
}

main() {
  cd "$REPO_ROOT"
  verify_source
  wait_for_gpu3
  verify_cuda_mask
  rm -f "$STATUS_DIR/status.failed" "$STATUS_DIR/status.complete"
  for seed in seed_k seed_l seed_m seed_n seed_o; do
    run_train_if_needed "$seed"
    for suffix in \
      history_only_action_value_only \
      history_only_residualize_action_value \
      history_composition_only \
      action_value_history_composition_only \
      residualize_action_value_history_composition; do
      run_probe_if_needed "$seed" "$suffix"
    done
  done
  run_step "$PYTHON_BIN" "$REPO_ROOT/scripts/build_k800_remote_manifest.py" \
    --repo-root "$REPO_ROOT" \
    --output "$STATUS_DIR/remote_manifest.json" \
    --source-revision "$SOURCE_REVISION" \
    --physical-gpu-index 3 \
    --commands-log "$RUN_LOG"
  printf '%s\n' "complete at $(date -Iseconds)" >"$STATUS_DIR/status.complete"
  log "all five runs and G5-G7 probe inputs complete"
}

main "$@"

