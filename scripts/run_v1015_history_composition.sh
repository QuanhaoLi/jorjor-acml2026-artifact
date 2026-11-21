#!/usr/bin/env bash

set -euo pipefail

BASE_COMMIT="${BASE_COMMIT:-4a0a8e6}"
COMMIT="${COMMIT:-}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/root/jorjor_runs}"
VENVDIR="${VENVDIR:-$ARCHIVE_ROOT/_venv_store/torch-cu124}"
BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260602}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-2048}"
GPU_UTIL_LIMIT="${GPU_UTIL_LIMIT:-20}"

log() {
  printf '[v1.0.15] %s at %s\n' "$*" "$(date -Iseconds)"
}

wait_for_gpu() {
  while true; do
    read -r used util < <(
      nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
        head -n 1 |
        tr -d ' ' |
        awk -F, '{print $1, $2}'
    )
    if [ "${used:-999999}" -lt "$GPU_MEMORY_LIMIT_MIB" ] && [ "${util:-999999}" -lt "$GPU_UTIL_LIMIT" ]; then
      log "gpu ready used=${used}MiB util=${util}%"
      return 0
    fi
    log "waiting for gpu used=${used:-unknown}MiB util=${util:-unknown}%"
    sleep 60
  done
}

run_step() {
  log "start $*"
  "$@"
  log "done $*"
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

link_reference_artifacts() {
  for seed in seed_k seed_l seed_m; do
    train_dir="outputs/limit_ar_action_value_only_${seed}_50k_4layer"
    if [ ! -e "$train_dir" ]; then
      ln -s "$ARCHIVE_ROOT/$BASE_COMMIT/$train_dir" "$train_dir"
    fi
    for suffix in history_only_action_value_only history_only_residualize_action_value; do
      probe_dir="outputs/probes/limit_ar_action_value_only_${seed}_4layer_range_${suffix}_50k"
      if [ ! -e "$probe_dir" ]; then
        ln -s "$ARCHIVE_ROOT/$BASE_COMMIT/$probe_dir" "$probe_dir"
      fi
    done
  done
}

require_reference_artifacts() {
  for seed in seed_k seed_l seed_m; do
    test -f "outputs/limit_ar_action_value_only_${seed}_50k_4layer/metrics.json"
    test -f "outputs/limit_ar_action_value_only_${seed}_50k_4layer/model.pt"
    test -f "outputs/probes/limit_ar_action_value_only_${seed}_4layer_range_history_only_action_value_only_50k/probe_metrics.json"
    test -f "outputs/probes/limit_ar_action_value_only_${seed}_4layer_range_history_only_action_value_only_50k/probe_correctness.jsonl"
    test -f "outputs/probes/limit_ar_action_value_only_${seed}_4layer_range_history_only_residualize_action_value_50k/probe_metrics.json"
    test -f "outputs/probes/limit_ar_action_value_only_${seed}_4layer_range_history_only_residualize_action_value_50k/probe_correctness.jsonl"
  done
}

run_history_composition_probes() {
  local suffix
  for seed in seed_k seed_l seed_m; do
    for control in \
      shuffle_history_order_seed_1 \
      shuffle_history_order_seed_2 \
      shuffle_history_order_seed_3 \
      reverse_history_order \
      sort_history_actions \
      history_presence_only \
      history_last_action_only; do
      for feature in residualize_action_value action_value_only; do
        suffix="${control}_${feature}"
        run_probe_if_missing "$seed" "$suffix"
      done
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
  require_reference_artifacts
  wait_for_gpu
  run_history_composition_probes
  run_step "$VENVDIR/bin/python" scripts/build_no_range_history_composition_pack.py \
    --archive-root "$ARCHIVE_ROOT" \
    --output-dir outputs/tables \
    --commit "$COMMIT" \
    --bootstrap-count "$BOOTSTRAP_COUNT" \
    --bootstrap-seed "$BOOTSTRAP_SEED"
}

main "$@"
