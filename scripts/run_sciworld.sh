#!/usr/bin/env bash
# Training only; install dependencies and provide model weights separately.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
ENGINE="${1:-vllm}"
if (( $# > 0 )); then shift; fi
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
: "${MODEL_PATH:?Set MODEL_PATH to a local model directory or public model ID}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SIZE="${TRAIN_SIZE:-16}"
GROUP_SIZE="${GROUP_SIZE:-8}"
N_GPUS="${N_GPUS:-8}"
VAL_SIZE="${VALIDATION_SIZE:-128}"
ENV_NAME="${ENV_NAME:-sciworld}"
PROJECT_NAME="${PROJECT_NAME:-env-as-scaffold}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_qwen3_8b_sciworld}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/text/sciworld}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/outputs/$PROJECT_NAME/$EXPERIMENT_NAME}"
mkdir -p "$SAVE_DIR/checkpoints"
export WANDB_DIR="$SAVE_DIR/wandb"
mkdir -p "$WANDB_DIR"
# These rows schedule environment tasks; no external preprocessing dataset is used.
"$PYTHON_BIN" scripts/prepare_data.py --output-dir "$DATA_DIR" \
  --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  "data.train_files=$DATA_DIR/train.parquet" \
  "data.val_files=$DATA_DIR/test.parquet" \
  "data.train_batch_size=$TRAIN_SIZE" \
  "data.val_batch_size=$VAL_SIZE" \
  data.max_prompt_length=6000 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=right \
  data.image_key=images \
  data.return_raw_chat=True \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL_SIZE:-2}" \
  "actor_rollout_ref.rollout.name=$ENGINE" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.max_thinking_budget=800 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=False \
  "env.env_name=$ENV_NAME" \
  env.history_length=2 \
  env.seed=0 \
  env.max_steps=15 \
  "env.rollout.n=$GROUP_SIZE" \
  "env.resources_per_worker.num_cpus=${NUM_CPUS_PER_ENV_WORKER:-0.01}" \
  trainer.critic_warmup=0 \
  'trainer.logger=[console]' \
  "trainer.project_name=$PROJECT_NAME" \
  "trainer.experiment_name=$EXPERIMENT_NAME" \
  "trainer.n_gpus_per_node=$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=5 \
  trainer.total_epochs=300 \
  trainer.val_before_train=False \
  "trainer.rollout_data_dir=$SAVE_DIR/rollout_data_dir" \
  "trainer.validation_data_dir=$SAVE_DIR/validation_data_dir" \
  trainer.log_val_generations=16 \
  "trainer.default_local_dir=$SAVE_DIR/checkpoints" \
  'actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_model]' \
  "$@"
