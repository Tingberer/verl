#!/usr/bin/env python3
"""
混合奖励管理器使用示例：Rule-based + 双奖励模型

演示如何结合：
1. 规则奖励（自定义函数）
2. 话术拟人打分模型 
3. 回复正确性模型

确保先导入 hybrid_reward_manager.py
"""

# 1. 导入自定义 HybridRewardManager
import sys
sys.path.append('/path/to/your/hybrid_reward_manager')  # 修改为实际路径
from hybrid_reward_manager import HybridRewardManager

# 2. 创建自定义规则奖励函数示例
def my_custom_reward_function(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    示例自定义规则奖励函数
    
    Args:
        data_source: 数据源标识
        solution_str: 模型生成的回答
        ground_truth: 标准答案
        extra_info: 额外信息
        
    Returns:
        float: 奖励分数 (0.0 - 1.0)
    """
    score = 0.0
    
    # 规则1: 长度奖励 - 回答不能太短也不能太长
    response_length = len(solution_str.split())
    if 10 <= response_length <= 100:
        score += 0.3
    elif response_length < 10:
        score += 0.1  # 太短扣分
    
    # 规则2: 关键词奖励 - 根据数据源检查关键词
    if data_source == "gsm8k":
        # 数学题应该包含计算过程
        if any(word in solution_str.lower() for word in ["calculate", "compute", "=", "answer"]):
            score += 0.4
    elif data_source == "general_qa":
        # 一般问答应该礼貌和完整
        if any(word in solution_str.lower() for word in ["please", "thank", "help", "certainly"]):
            score += 0.2
    
    # 规则3: 正确性检查（如果有标准答案）
    if ground_truth and solution_str.lower().strip() == ground_truth.lower().strip():
        score += 0.3
    
    return min(score, 1.0)  # 确保不超过1.0

# 3. 配置参数示例
def setup_hybrid_reward_config():
    """设置混合奖励配置"""
    return {
        # 启用奖励模型
        "reward_model.enable": True,
        
        # 使用混合奖励管理器
        "reward_model.reward_manager": "hybrid",
        
        # 三个奖励组件的权重 (总和必须为1.0)
        "reward_model.weight_rule": 0.3,              # 规则奖励权重
        "reward_model.weight_anthropomorphism": 0.4,   # 话术拟人权重
        "reward_model.weight_correctness": 0.3,        # 正确性权重
        
        # 神经网络模型路径
        "reward_model.anthropomorphism_model_path": "~/models/anthropomorphism_reward_model",
        "reward_model.correctness_model_path": "~/models/correctness_reward_model",
        
        # 自定义规则函数配置
        "custom_reward_function.path": "my_custom_reward.py",
        "custom_reward_function.name": "my_custom_reward_function",
        
        # 其他设置
        "reward_model.use_trust_remote_code": False,
        "reward_model.device": "cuda",
        
        # 训练配置
        "data.train_batch_size": 1024,
        "trainer.total_epochs": 10,
    }

# 4. 训练脚本配置示例
def create_hybrid_training_script():
    """生成完整训练脚本"""
    script = '''#!/bin/bash

# 混合奖励训练脚本：Rule-based + 话术拟人 + 正确性
set -x

# 数据路径
train_files="['/path/to/train.parquet']"
test_files="['/path/to/test.parquet']"

# 模型路径配置
ANTHROPOMORPHISM_MODEL="$HOME/models/anthropomorphism_reward_model"
CORRECTNESS_MODEL="$HOME/models/correctness_reward_model"

# 三重奖励权重配置 (总和 = 1.0)
WEIGHT_RULE=0.3              # 规则奖励
WEIGHT_ANTHROPOMORPHISM=0.4  # 话术拟人
WEIGHT_CORRECTNESS=0.3       # 正确性

echo "🎯 混合奖励配置："
echo "  规则奖励权重: $WEIGHT_RULE"
echo "  话术拟人权重: $WEIGHT_ANTHROPOMORPHISM" 
echo "  正确性权重: $WEIGHT_CORRECTNESS"

# 确保导入自定义管理器
export PYTHONPATH="$PYTHONPATH:$(dirname $0)"

python3 -m verl.trainer.main_ppo \\
    algorithm.adv_estimator=grpo \\
    data.train_files="$train_files" \\
    data.val_files="$test_files" \\
    data.train_batch_size=1024 \\
    data.max_prompt_length=1024 \\
    data.max_response_length=512 \\
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    reward_model.enable=True \\
    reward_model.reward_manager=hybrid \\
    reward_model.weight_rule=$WEIGHT_RULE \\
    reward_model.weight_anthropomorphism=$WEIGHT_ANTHROPOMORPHISM \\
    reward_model.weight_correctness=$WEIGHT_CORRECTNESS \\
    reward_model.anthropomorphism_model_path="$ANTHROPOMORPHISM_MODEL" \\
    reward_model.correctness_model_path="$CORRECTNESS_MODEL" \\
    custom_reward_function.path=my_custom_reward.py \\
    custom_reward_function.name=my_custom_reward_function \\
    trainer.logger='["console","wandb"]' \\
    trainer.project_name=hybrid_reward_experiment \\
    trainer.experiment_name=qwen2_7b_hybrid_rule_anthro_correct \\
    trainer.n_gpus_per_node=8 \\
    trainer.total_epochs=15 $@
'''
    return script

# 5. 权重调优建议
def suggest_weight_configurations():
    """不同场景的权重配置建议"""
    scenarios = {
        "数学推理任务": {
            "weight_rule": 0.5,          # 规则很重要（步骤、格式）
            "weight_anthropomorphism": 0.2,  # 拟人化次要
            "weight_correctness": 0.3,   # 正确性重要
            "description": "强调逻辑推理和计算正确性"
        },
        
        "客服对话任务": {
            "weight_rule": 0.2,          # 规则次要
            "weight_anthropomorphism": 0.6,  # 拟人化最重要
            "weight_correctness": 0.2,   # 正确性次要
            "description": "强调对话体验和情感表达"
        },
        
        "知识问答任务": {
            "weight_rule": 0.3,          # 规则中等
            "weight_anthropomorphism": 0.2,  # 拟人化次要  
            "weight_correctness": 0.5,   # 正确性最重要
            "description": "强调准确性和可靠性"
        },
        
        "创意写作任务": {
            "weight_rule": 0.4,          # 规则重要（格式、风格）
            "weight_anthropomorphism": 0.4,  # 拟人化重要
            "weight_correctness": 0.2,   # 正确性次要
            "description": "平衡创意和表达质量"
        }
    }
    
    return scenarios

if __name__ == "__main__":
    print("🔧 混合奖励管理器配置示例")
    print("=" * 60)
    
    # 显示配置
    config = setup_hybrid_reward_config()
    print("📋 基础配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    
    # 显示训练脚本
    script = create_hybrid_training_script()
    print("📜 训练脚本示例:")
    print(script[:500] + "..." if len(script) > 500 else script)
    
    print("\n" + "=" * 60)
    
    # 显示权重配置建议
    scenarios = suggest_weight_configurations()
    print("💡 不同场景权重配置建议:")
    for scenario, config in scenarios.items():
        print(f"\n  📌 {scenario}:")
        print(f"     规则权重: {config['weight_rule']}")
        print(f"     拟人权重: {config['weight_anthropomorphism']}")
        print(f"     正确性权重: {config['weight_correctness']}")
        print(f"     说明: {config['description']}")
    
    print("\n🚀 使用提示:")
    print("  1. 创建自定义规则函数文件 (my_custom_reward.py)")
    print("  2. 修改模型路径和权重配置")
    print("  3. 运行训练脚本")
    print("  4. 根据验证结果调整权重")