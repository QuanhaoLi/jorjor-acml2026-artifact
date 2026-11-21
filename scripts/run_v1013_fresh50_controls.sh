#!/usr/bin/env bash

set -euo pipefail

BASE_COMMIT="${BASE_COMMIT:-4a0a8e6}"
COMMIT_50K="${COMMIT_50K:-}"
COMMIT_10K="${COMMIT_10K:-9a368b2}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/root/jorjor_runs}"
VENVDIR="${VENVDIR:-$ARCHIVE_ROOT/_venv_store/torch-cu124}"
BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260602}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-2048}"
GPU_UTIL_LIMIT="${GPU_UTIL_LIMIT:-20}"

log() {
  printf '[v1.0.13] %s at %s\n' "$*" "$(date -Iseconds)"
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

link_base_artifacts() {
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

require_base_artifacts() {
  for seed in seed_k seed_l seed_m; do
    test -f "outputs/limit_ar_action_value_only_${seed}_50k_4layer/metrics.json"
    test -f "outputs/limit_ar_action_value_only_${seed}_50k_4layer/model.pt"
    test -f "outputs/probes/limit_ar_action_value_only_${seed}_4layer_range_history_only_action_value_only_50k/probe_metrics.json"
    test -f "outputs/probes/limit_ar_action_value_only_${seed}_4layer_range_history_only_residualize_action_value_50k/probe_metrics.json"
  done
  test -d "$ARCHIVE_ROOT/$COMMIT_10K/outputs/probes"
}

run_supplemental_probes() {
  local suffix
  for seed in seed_k seed_l seed_m; do
    for suffix in \
      history_only_action_value_only_repeat_1 \
      history_only_action_value_only_repeat_2 \
      history_only_action_value_only_repeat_3 \
      history_only_residualize_action_value_repeat_1 \
      history_only_residualize_action_value_repeat_2 \
      history_only_residualize_action_value_repeat_3 \
      history_only_residualize_action_value_shuffle_global_labels \
      history_only_residualize_action_value_shuffle_within_bucket_labels \
      drop_history \
      shuffle_history_order \
      bag_of_actions \
      no_observation \
      round_position_action_prior \
      history_only_action_only \
      drop_history_residualize_action_value \
      shuffle_history_order_residualize_action_value \
      bag_of_actions_residualize_action_value \
      no_observation_residualize_action_value \
      drop_history_action_value_only \
      shuffle_history_order_action_value_only \
      bag_of_actions_action_value_only \
      no_observation_action_value_only; do
      run_step "$VENVDIR/bin/python" -m jorjor probe-limit-ar \
        --config "configs/probe/limit_ar_action_value_only_${seed}_4layer_range_${suffix}_50k.json"
    done
  done
}

main() {
  if [ -z "$COMMIT_50K" ]; then
    printf 'COMMIT_50K must be set for archive-extracted runs\n' >&2
    exit 2
  fi
  cd "$ARCHIVE_ROOT/$COMMIT_50K"
  mkdir -p outputs/probes outputs/tables
  ln -sfn "$VENVDIR" .venv
  link_base_artifacts
  require_base_artifacts
  wait_for_gpu
  run_supplemental_probes
  run_step "$VENVDIR/bin/python" scripts/build_no_range_locked_layer_pack.py \
    --archive-root "$ARCHIVE_ROOT" \
    --output-dir outputs/tables \
    --commit "$COMMIT_50K"
  run_step "$VENVDIR/bin/python" scripts/build_no_range_bootstrap_pack.py \
    --archive-root "$ARCHIVE_ROOT" \
    --output-dir outputs/tables \
    --commit "$COMMIT_50K" \
    --bootstrap-count "$BOOTSTRAP_COUNT" \
    --bootstrap-seed "$BOOTSTRAP_SEED"
  run_step "$VENVDIR/bin/python" scripts/build_no_range_fresh50_control_pack.py \
    --archive-root "$ARCHIVE_ROOT" \
    --output-dir outputs/tables \
    --commit-50k "$COMMIT_50K" \
    --commit-10k "$COMMIT_10K" \
    --bootstrap-count "$BOOTSTRAP_COUNT" \
    --bootstrap-seed "$BOOTSTRAP_SEED"
}

main "$@"
