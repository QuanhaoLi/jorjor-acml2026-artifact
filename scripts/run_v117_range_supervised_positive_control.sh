#!/usr/bin/env bash

set -euo pipefail

BASE_COMMIT="${BASE_COMMIT:-8a7a72b}"
SEED_D_COMMIT="${SEED_D_COMMIT:-8502bf8}"
COMMIT="${COMMIT:-}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/root/jorjor_runs}"
VENVDIR="${VENVDIR:-$ARCHIVE_ROOT/_venv_store/torch-cu124}"
BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260603}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-2048}"
GPU_UTIL_LIMIT="${GPU_UTIL_LIMIT:-20}"
GPU_WAIT_BACKEND="${GPU_WAIT_BACKEND:-auto}"

log() {
  printf '[v1.17] %s at %s\n' "$*" "$(date -Iseconds)"
}

wait_for_gpu() {
  local backend="$GPU_WAIT_BACKEND"
  if [ "$backend" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      backend="nvidia"
    elif command -v rocm-smi >/dev/null 2>&1; then
      backend="rocm"
    else
      backend="none"
    fi
  fi
  if [ "$backend" = "none" ]; then
    log "gpu wait disabled"
    return 0
  fi
  while true; do
    if [ "$backend" = "nvidia" ]; then
      read -r used util < <(
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
          head -n 1 |
          tr -d ' ' |
          awk -F, '{print $1, $2}'
      )
    elif [ "$backend" = "rocm" ]; then
      used="$(rocm-smi --showmemuse --csv 2>/dev/null | awk -F, 'NR==2 {gsub(/[^0-9.]/, "", $NF); print int($NF)}')"
      util="$(rocm-smi --showuse --csv 2>/dev/null | awk -F, 'NR==2 {gsub(/[^0-9.]/, "", $NF); print int($NF)}')"
    else
      printf 'unknown GPU_WAIT_BACKEND=%s\n' "$backend" >&2
      exit 2
    fi
    if [ "${used:-999999}" -lt "$GPU_MEMORY_LIMIT_MIB" ] && [ "${util:-999999}" -lt "$GPU_UTIL_LIMIT" ]; then
      log "gpu ready backend=${backend} used=${used}MiB util=${util}%"
      return 0
    fi
    log "waiting for gpu backend=${backend} used=${used:-unknown}MiB util=${util:-unknown}%"
    sleep 60
  done
}

run_step() {
  log "start $*"
  "$@"
  log "done $*"
}

source_commit_for_seed() {
  local seed="$1"
  if [ "$seed" = "seed_d" ]; then
    printf '%s\n' "$SEED_D_COMMIT"
  else
    printf '%s\n' "$BASE_COMMIT"
  fi
}

link_reference_artifacts() {
  local seed source_commit train_dir probe_dir suffix
  for seed in seed_d seed_f seed_g; do
    source_commit="$(source_commit_for_seed "$seed")"
    train_dir="outputs/limit_ar_teacher_${seed}_50k_4layer"
    if [ ! -e "$train_dir" ]; then
      ln -s "$ARCHIVE_ROOT/$source_commit/$train_dir" "$train_dir"
    fi
    for suffix in history_only_action_value_only history_only_residualize_action_value; do
      probe_dir="outputs/probes/limit_ar_teacher_${seed}_4layer_range_${suffix}_50k"
      if [ ! -e "$probe_dir" ] && [ -e "$ARCHIVE_ROOT/$source_commit/$probe_dir" ]; then
        ln -s "$ARCHIVE_ROOT/$source_commit/$probe_dir" "$probe_dir"
      fi
    done
  done
}

run_probe_if_missing_correctness() {
  local seed="$1"
  local suffix="$2"
  local probe_dir="outputs/probes/limit_ar_teacher_${seed}_4layer_range_${suffix}_50k"
  if [ -f "$probe_dir/probe_metrics.json" ] && [ -f "$probe_dir/probe_correctness.jsonl" ]; then
    log "skip existing $probe_dir"
    return 0
  fi
  if [ -L "$probe_dir" ]; then
    rm "$probe_dir"
  fi
  run_step "$VENVDIR/bin/python" -m jorjor probe-limit-ar \
    --config "configs/probe/limit_ar_teacher_${seed}_4layer_range_${suffix}_50k.json"
}

require_train_artifacts() {
  local seed
  for seed in seed_d seed_f seed_g; do
    test -f "outputs/limit_ar_teacher_${seed}_50k_4layer/metrics.json"
    test -f "outputs/limit_ar_teacher_${seed}_50k_4layer/model.pt"
  done
}

run_positive_control_probes() {
  local seed suffix
  for seed in seed_d seed_f seed_g; do
    for suffix in \
      history_only_action_value_only \
      history_only_residualize_action_value \
      action_value_history_composition_only \
      residualize_action_value_history_composition; do
      run_probe_if_missing_correctness "$seed" "$suffix"
    done
  done
}

main() {
  if [ -z "$COMMIT" ]; then
    printf 'COMMIT must be set for archive-extracted runs\n' >&2
    exit 2
  fi
  cd "$ARCHIVE_ROOT/$COMMIT"
  mkdir -p outputs/probes outputs/tables
  ln -sfn "$VENVDIR" .venv
  link_reference_artifacts
  require_train_artifacts
  wait_for_gpu
  run_positive_control_probes
  run_step "$VENVDIR/bin/python" scripts/build_range_supervised_positive_control_pack.py \
    --archive-root "$ARCHIVE_ROOT" \
    --output-dir outputs/tables \
    --commit "$COMMIT" \
    --bootstrap-count "$BOOTSTRAP_COUNT" \
    --bootstrap-seed "$BOOTSTRAP_SEED"
}

main "$@"
