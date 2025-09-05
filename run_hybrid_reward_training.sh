#!/bin/bash

# 混合奖励模型训练脚本
# Rule-based + 话术拟人打分模型 + 回复正确性模型

set -x

# =============================================================================
# 配置区域 - 根据你的环境修改以下路径和参数
# =============================================================================

# 数据路径配置
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

# 神经网络奖励模型路径 (请根据实际情况修改)
ANTHROPOMORPHISM_MODEL_PATH="$HOME/models/anthropomorphism_reward_model"
CORRECTNESS_MODEL_PATH="$HOME/models/correctness_reward_model"

# 三重奖励权重配置 (总和必须为1.0)
WEIGHT_RULE=0.3              # 规则奖励权重
WEIGHT_ANTHROPOMORPHISM=0.4  # 话术拟人权重  
WEIGHT_CORRECTNESS=0.3       # 正确性权重

# 验证权重总和
TOTAL_WEIGHT=$(echo "$WEIGHT_RULE + $WEIGHT_ANTHROPOMORPHISM + $WEIGHT_CORRECTNESS" | bc)
if [ "$TOTAL_WEIGHT" != "1.0" ]; then
    echo "❌ 错误: 权重总和必须为1.0，当前为 $TOTAL_WEIGHT"
    exit 1
fi

# 实验名称
EXPERIMENT_NAME="hybrid_reward_rule${WEIGHT_RULE}_anthro${WEIGHT_ANTHROPOMORPHISM}_correct${WEIGHT_CORRECTNESS}"

echo "🎯 混合奖励模型训练配置："
echo "  规则奖励: 权重 $WEIGHT_RULE"
echo "  话术拟人模型: $ANTHROPOMORPHISM_MODEL_PATH (权重: $WEIGHT_ANTHROPOMORPHISM)"
echo "  正确性模型: $CORRECTNESS_MODEL_PATH (权重: $WEIGHT_CORRECTNESS)"
echo "  实验名称: $EXPERIMENT_NAME"

# =============================================================================
# 环境设置
# =============================================================================

# 确保导入自定义 HybridRewardManager 和规则函数
export PYTHONPATH="$PYTHONPATH:$(dirname $0)"

# 检查必要文件是否存在
if [ ! -f "hybrid_reward_manager.py" ]; then
    echo "❌ 错误: 找不到 hybrid_reward_manager.py"
    exit 1
fi

if [ ! -f "my_custom_reward.py" ]; then
    echo "❌ 错误: 找不到 my_custom_reward.py (自定义规则函数)"
    exit 1
fi

echo "✅ 环境检查通过"

# =============================================================================
# 训练执行
# =============================================================================

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
    reward_model.reward_manager=hybrid \
    reward_model.weight_rule=$WEIGHT_RULE \
    reward_model.weight_anthropomorphism=$WEIGHT_ANTHROPOMORPHISM \
    reward_model.weight_correctness=$WEIGHT_CORRECTNESS \
    reward_model.anthropomorphism_model_path="$ANTHROPOMORPHISM_MODEL_PATH" \
    reward_model.correctness_model_path="$CORRECTNESS_MODEL_PATH" \
    reward_model.use_trust_remote_code=False \
    reward_model.device=cuda \
    reward_model.micro_batch_size_per_gpu=32 \
    custom_reward_function.path=my_custom_reward.py \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='hybrid_reward_verl' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@

# =============================================================================
# 训练后处理
# =============================================================================

if [ $? -eq 0 ]; then
    echo "✅ 混合奖励模型训练成功完成！"
    echo "📊 实验结果保存在: $EXPERIMENT_NAME"
    echo "🔍 权重配置总结:"
    echo "   - 规则奖励: ${WEIGHT_RULE} (30%)"
    echo "   - 话术拟人: ${WEIGHT_ANTHROPOMORPHISM} (40%)"  
    echo "   - 回复正确: ${WEIGHT_CORRECTNESS} (30%)"
else
    echo "❌ 训练失败，请检查错误信息"
    exit 1
fi

# 可选: 自动评估和分析
echo ""
echo "💡 下一步建议:"
echo "1. 查看 wandb 日志分析各组件贡献"
echo "2. 根据验证结果调整权重比例"  
echo "3. 尝试不同的规则函数或模型组合"
echo ""