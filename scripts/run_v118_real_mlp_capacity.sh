#!/usr/bin/env bash

set -euo pipefail

COMMIT="${COMMIT:-}"
LINEAR_COMMIT="${LINEAR_COMMIT:-2dfcb42}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/root/jorjor_runs}"
VENVDIR="${VENVDIR:-$ARCHIVE_ROOT/_venv_store/torch-cu124}"
BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260603}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-2048}"
GPU_UTIL_LIMIT="${GPU_UTIL_LIMIT:-20}"
GPU_WAIT_BACKEND="${GPU_WAIT_BACKEND:-auto}"

log() {
  printf '[v1.18] %s at %s\n' "$*" "$(date -Iseconds)"
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

link_linear_artifacts() {
  local seed train_dir probe_dir suffix
  for seed in seed_k seed_l seed_m; do
    train_dir="outputs/limit_ar_action_value_only_${seed}_50k_4layer"
    if [ ! -e "$train_dir" ]; then
      ln -s "$ARCHIVE_ROOT/$LINEAR_COMMIT/$train_dir" "$train_dir"
    fi
    for suffix in history_only_action_value_only action_value_history_composition_only; do
      probe_dir="outputs/probes/limit_ar_action_value_only_${seed}_4layer_range_${suffix}_50k"
      if [ ! -e "$probe_dir" ]; then
        ln -s "$ARCHIVE_ROOT/$LINEAR_COMMIT/$probe_dir" "$probe_dir"
      fi
    done
  done
}

require_linear_artifacts() {
  local seed suffix probe_dir
  for seed in seed_k seed_l seed_m; do
    test -f "outputs/limit_ar_action_value_only_${seed}_50k_4layer/metrics.json"
    test -f "outputs/limit_ar_action_value_only_${seed}_50k_4layer/model.pt"
    for suffix in history_only_action_value_only action_value_history_composition_only; do
      probe_dir="outputs/probes/limit_ar_action_value_only_${seed}_4layer_range_${suffix}_50k"
      test -f "$probe_dir/probe_metrics.json"
      test -f "$probe_dir/probe_correctness.jsonl"
    done
  done
}

run_probe_if_missing() {
  local seed="$1"
  local suffix="$2"
  local probe_dir="outputs/probes/limit_ar_action_value_only_${seed}_4layer_range_${suffix}_50k"
  if [ -f "$probe_dir/probe_metrics.json" ] && [ -f "$probe_dir/probe_correctness.jsonl" ]; then
    log "skip existing $probe_dir"
    return 0
  fi
  run_step "$VENVDIR/bin/python" -m jorjor probe-limit-ar \
    --config "configs/probe/limit_ar_action_value_only_${seed}_4layer_range_${suffix}_50k.json"
}

run_mlp_probes() {
  local seed suffix
  for seed in seed_k seed_l seed_m; do
    for suffix in \
      history_only_residualize_action_value_mlp \
      residualize_action_value_history_composition_mlp; do
      run_probe_if_missing "$seed" "$suffix"
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
  link_linear_artifacts
  require_linear_artifacts
  wait_for_gpu
  run_mlp_probes
  run_step "$VENVDIR/bin/python" scripts/build_probe_capacity_s6_s7_report.py \
    --archive-root "$ARCHIVE_ROOT" \
    --output-dir outputs/tables \
    --mlp-commit "$COMMIT" \
    --linear-commit "$LINEAR_COMMIT" \
    --bootstrap-count "$BOOTSTRAP_COUNT" \
    --bootstrap-seed "$BOOTSTRAP_SEED"
}

main "$@"
