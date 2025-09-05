#!/usr/bin/env python3
"""
使用示例：如何在 verl 中使用 DualRewardManager 进行双奖励模型训练

确保先导入 dual_reward_manager.py，然后按以下方式配置训练脚本
"""

# 1. 首先确保导入 dual_reward_manager
import sys
sys.path.append('/path/to/your/dual_reward_manager')  # 修改为实际路径
from dual_reward_manager import DualRewardManager

# 2. 在训练脚本中设置双奖励管理器
def setup_dual_reward_training():
    # 示例配置参数
    config_overrides = {
        # 启用奖励模型
        "reward_model.enable": True,
        
        # 使用自定义的双奖励管理器
        "reward_model.reward_manager": "dual",
        
        # 双模型路径
        "reward_model.anthropomorphism_model_path": "~/models/anthropomorphism_reward_model",
        "reward_model.correctness_model_path": "~/models/correctness_reward_model", 
        
        # 权重配置 (权重和必须为 1.0)
        "reward_model.weight_anthropomorphism": 0.6,  # 话术拟人权重
        "reward_model.weight_correctness": 0.4,       # 正确性权重
        
        # 其他设置
        "reward_model.use_trust_remote_code": False,
        "reward_model.device": "cuda",
        
        # 标准 verl 配置...
        "data.train_batch_size": 1024,
        "trainer.total_epochs": 10,
    }
    
    return config_overrides

# 3. 完整的训练命令示例
training_command = """
# 方式一：命令行参数
python3 -m verl.trainer.main_ppo \\
    algorithm.adv_estimator=grpo \\
    data.train_files="['/path/to/train.parquet']" \\
    data.val_files="['/path/to/val.parquet']" \\
    data.train_batch_size=1024 \\
    data.max_prompt_length=1024 \\
    data.max_response_length=512 \\
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    reward_model.enable=True \\
    reward_model.reward_manager=dual \\
    reward_model.anthropomorphism_model_path=~/models/anthropomorphism_reward_model \\
    reward_model.correctness_model_path=~/models/correctness_reward_model \\
    reward_model.weight_anthropomorphism=0.6 \\
    reward_model.weight_correctness=0.4 \\
    trainer.logger='["console","wandb"]' \\
    trainer.project_name=dual_reward_experiment \\
    trainer.experiment_name=qwen2_7b_dual_reward \\
    trainer.n_gpus_per_node=8 \\
    trainer.nnodes=1 \\
    trainer.total_epochs=15
"""

print("双奖励模型训练配置示例：")
print(training_command)

# 4. 如果需要程序化配置
if __name__ == "__main__":
    # 导入必要的模块
    from verl.trainer.main_ppo import main
    
    # 设置配置
    overrides = setup_dual_reward_training()
    
    print("🎯 双奖励模型配置:")
    for key, value in overrides.items():
        print(f"  {key}: {value}")
    
    # 启动训练
    # main(overrides)  # 取消注释以实际运行