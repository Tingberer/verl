#!/bin/bash

# 双奖励模型训练脚本示例
# 使用话术拟人打分模型 + 回复正确性模型

set -x

# 数据路径配置
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

# 双奖励模型路径 (请根据实际情况修改)
ANTHROPOMORPHISM_MODEL_PATH="$HOME/models/anthropomorphism_reward_model"
CORRECTNESS_MODEL_PATH="$HOME/models/correctness_reward_model"

# 权重配置 (可根据需要调整，确保总和为1.0)
WEIGHT_ANTHROPOMORPHISM=0.6
WEIGHT_CORRECTNESS=0.4

echo "🎯 双奖励模型训练配置："
echo "  话术拟人模型: $ANTHROPOMORPHISM_MODEL_PATH (权重: $WEIGHT_ANTHROPOMORPHISM)"
echo "  正确性模型: $CORRECTNESS_MODEL_PATH (权重: $WEIGHT_CORRECTNESS)"

# 确保导入自定义 DualRewardManager
export PYTHONPATH="$PYTHONPATH:$(dirname $0)"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=True \
    reward_model.reward_manager=dual \
    reward_model.anthropomorphism_model_path="$ANTHROPOMORPHISM_MODEL_PATH" \
    reward_model.correctness_model_path="$CORRECTNESS_MODEL_PATH" \
    reward_model.weight_anthropomorphism=$WEIGHT_ANTHROPOMORPHISM \
    reward_model.weight_correctness=$WEIGHT_CORRECTNESS \
    reward_model.use_trust_remote_code=False \
    reward_model.device=cuda \
    reward_model.micro_batch_size_per_gpu=32 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='dual_reward_verl' \
    trainer.experiment_name="qwen2_7b_dual_reward_anthropomorphism_${WEIGHT_ANTHROPOMORPHISM}_correctness_${WEIGHT_CORRECTNESS}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@

echo "✅ 双奖励模型训练完成！"