#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
BENCHMARK="${1:-sciworld}"
ALGORITHM="${2:-grpo}"
FEEDBACK="${3:-enriched}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
case "$BENCHMARK" in
  sciworld) ENV_NAME=sciworld; VAL_SIZE=128; MAX_STEPS=15 ;;
  bfcl) ENV_NAME=bfcl; VAL_SIZE=400; MAX_STEPS=50 ;;
  *) echo 'Benchmark must be sciworld or bfcl' >&2; exit 2 ;;
esac
case "$FEEDBACK" in
  standard) ;;
  enriched)
    if [[ "$BENCHMARK" == sciworld ]]; then ENV_NAME=sciworld_diversev7; else ENV_NAME=bfcl_diversev3; fi ;;
  inconsistent)
    [[ "$BENCHMARK" == sciworld ]] || { echo 'inconsistent is SciWorld-only' >&2; exit 2; }
    ENV_NAME=sciworld_diversev8 ;;
  *) echo 'Feedback must be standard, enriched or inconsistent' >&2; exit 2 ;;
esac
ALG_ARGS=(actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean)
case "$ALGORITHM" in
  grpo) ;;
  dapo) ALG_ARGS=(actor_rollout_ref.actor.loss_agg_mode=token-mean actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 algorithm.filter_groups.enable=True) ;;
  gspo) ALG_ARGS+=(actor_rollout_ref.actor.policy_loss.loss_mode=gspo actor_rollout_ref.actor.clip_ratio_low=0.0003 actor_rollout_ref.actor.clip_ratio_high=0.0004) ;;
  *) echo 'Algorithm must be grpo, dapo or gspo' >&2; exit 2 ;;
esac
: "${MODEL_PATH:?Set MODEL_PATH to a local model directory or public model ID}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SIZE="${TRAIN_SIZE:-16}"
VAL_SIZE="${VALIDATION_SIZE:-$VAL_SIZE}"
GROUP_SIZE="${GROUP_SIZE:-8}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/text/$BENCHMARK}"
"$PYTHON_BIN" scripts/prepare_data.py --output-dir "$DATA_DIR" --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE"
"$PYTHON_BIN" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  "data.train_files=$DATA_DIR/train.parquet" "data.val_files=$DATA_DIR/test.parquet" \
  "data.train_batch_size=$TRAIN_SIZE" "data.val_batch_size=$VAL_SIZE" \
  data.max_prompt_length=8192 data.max_response_length=2048 data.return_raw_chat=True \
  data.filter_overlong_prompts=False \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  "actor_rollout_ref.actor.ppo_mini_batch_size=$((TRAIN_SIZE * GROUP_SIZE))" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.use_invalid_action_penalty=False \
  actor_rollout_ref.rollout.name=vllm \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL_SIZE:-1}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  "env.env_name=$ENV_NAME" "env.max_steps=$MAX_STEPS" "env.rollout.n=$GROUP_SIZE" \
  env.seed=0 env.resources_per_worker.num_cpus=0.1 \
  "trainer.n_gpus_per_node=${N_GPUS:-8}" trainer.nnodes=1 \
  'trainer.logger=[console]' trainer.project_name=env-as-scaffold \
  "trainer.experiment_name=${BENCHMARK}_${ALGORITHM}_${FEEDBACK}" \
  trainer.total_epochs=200 trainer.total_training_steps=200 trainer.test_freq=5 \
  trainer.save_freq=50 trainer.val_before_train=True trainer.resume_mode=disable \
  "${ALG_ARGS[@]}" "$@"
