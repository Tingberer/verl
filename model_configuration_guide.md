# 模型配置指南：Qwen3 4B + DeepSeek-R1-Distill-Qwen-14B

本指南详细说明如何配置和使用基于 Qwen3 4B 的话术拟人打分模型和基于 DeepSeek-R1-Distill-Qwen-14B 的回复正确性模型。

## 📋 模型架构信息

### 🎭 话术拟人打分模型
- **基础架构**: Qwen3 4B
- **模型类型**: 序列分类模型 (AutoModelForSequenceClassification)
- **输出**: 单值奖励分数 (0.0 - 1.0)
- **用途**: 评估回答的拟人化程度、情感表达和交互质量

### ✅ 回复正确性模型
- **基础架构**: DeepSeek-R1-Distill-Qwen-14B
- **模型类型**: 序列分类模型 (AutoModelForSequenceClassification)
- **输出**: 单值奖励分数 (0.0 - 1.0)
- **用途**: 评估回答的事实准确性和逻辑正确性

## 🔧 自动化配置特性

我们的 DualRewardManager 和 HybridRewardManager 具备智能架构检测功能：

### 架构检测逻辑

```python
def _detect_model_architecture(self, config, model_type: str, local_path: str):
    # 自动检测 Qwen 系列模型
    if 'qwen' in architecture or 'qwen' in model_type.lower():
        # Qwen3 4B 话术拟人模型配置
        if model_type.lower() == 'anthropomorphism':
            return qwen3_4b_anthropomorphism_config
    
    # 自动检测 DeepSeek 系列模型
    elif 'deepseek' in architecture or 'deepseek' in model_type.lower():
        # DeepSeek-R1-Distill-Qwen-14B 正确性模型配置
        if model_type.lower() == 'correctness':
            return deepseek_r1_distill_qwen_14b_config
```

### 模型特定优化

#### Qwen3 4B 优化设置
- ✅ 禁用缓存 (`use_cache=False`) 节省内存
- ✅ 分类器 dropout 设置为 0.0
- ✅ 输出标签数设置为 1
- ✅ 使用 bfloat16 精度

#### DeepSeek-R1-Distill-Qwen-14B 优化设置  
- ✅ 低内存占用模式 (`low_cpu_mem_usage=True`)
- ✅ 手动设备映射管理
- ✅ 动态分类器头调整 (如果需要)
- ✅ 14B 模型内存优化

## 📁 模型文件结构要求

确保你的模型文件夹包含以下基本文件：

```
your_model_path/
├── config.json          # 模型配置文件
├── pytorch_model.bin     # 或 model.safetensors
├── tokenizer.json        # tokenizer 配置
├── tokenizer_config.json
└── special_tokens_map.json
```

## ⚙️ 使用配置示例

### 基本双奖励配置

```bash
python3 -m verl.trainer.main_ppo \
    reward_model.enable=True \
    reward_model.reward_manager=dual \
    reward_model.anthropomorphism_model_path=/path/to/qwen3-4b-anthropomorphism \
    reward_model.correctness_model_path=/path/to/deepseek-r1-distill-qwen-14b \
    reward_model.weight_anthropomorphism=0.6 \
    reward_model.weight_correctness=0.4 \
    # ... 其他配置
```

### 混合奖励配置 (Rule + Dual)

```bash
python3 -m verl.trainer.main_ppo \
    reward_model.enable=True \
    reward_model.reward_manager=hybrid \
    reward_model.weight_rule=0.3 \
    reward_model.weight_anthropomorphism=0.4 \
    reward_model.weight_correctness=0.3 \
    reward_model.anthropomorphism_model_path=/path/to/qwen3-4b-anthropomorphism \
    reward_model.correctness_model_path=/path/to/deepseek-r1-distill-qwen-14b \
    custom_reward_function.path=my_custom_reward.py \
    custom_reward_function.name=compute_score \
    # ... 其他配置
```

## 🚨 常见问题解决

### 问题 1: 模型加载失败
```
Error: Unexpected model output format for anthropomorphism
```

**解决方案**:
- 检查模型是否是正确的 SequenceClassification 格式
- 确保模型配置中 `num_labels=1`
- 检查模型路径是否正确

### 问题 2: 内存不足
```
CUDA out of memory
```

**解决方案**:
```python
# 启用内存优化选项
reward_model.device=cuda
reward_model.micro_batch_size_per_gpu=16  # 减小批次大小
reward_model.use_dynamic_bsz=True
```

### 问题 3: 推理速度慢
```
Training is too slow due to reward model inference
```

**解决方案**:
```python
# 并行化和优化设置
reward_model.launch_reward_fn_async=True
trainer.n_gpus_per_node=8  # 使用多GPU
reward_model.forward_max_token_len_per_gpu=8192
```

## 📊 性能调优建议

### 内存优化策略

1. **分层加载**: 话术模型(4B)放内存，正确性模型(14B)可以offload
2. **批次大小**: 根据GPU内存调整 `micro_batch_size_per_gpu`
3. **精度设置**: 使用 bfloat16 减少内存占用

### 计算效率优化

1. **模型并行**: 14B模型可以考虑tensor parallel
2. **异步推理**: 启用 `launch_reward_fn_async=True`
3. **缓存管理**: 合理设置 `use_cache` 参数

## 🔍 调试技巧

### 启用详细日志

```python
# 在代码中添加调试信息
print("📊 Model architectures:")
print(f"   Anthropomorphism: {self.anthropomorphism_model}")
print(f"   Correctness: {self.correctness_model}")
```

### 验证模型输出

```python
# 测试模型推理
test_input = "测试输入文本"
with torch.no_grad():
    output = model(test_input)
    print(f"模型输出形状: {output.logits.shape}")
    print(f"预测分数: {output.logits.squeeze().item()}")
```

## 🚀 最佳实践总结

1. **✅ 路径配置**: 使用绝对路径指定模型位置
2. **✅ 权重平衡**: 根据任务特点调整权重比例
3. **✅ 内存监控**: 定期检查GPU内存使用情况
4. **✅ 日志记录**: 启用详细日志便于问题排查
5. **✅ 渐进测试**: 先单模型测试，再组合使用

通过以上配置，你的双奖励模型系统应该能够稳定高效地运行！