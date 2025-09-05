# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

verl (Volcano Engine Reinforcement Learning) is a flexible, efficient RL training library for large language models. It supports multiple RL algorithms (PPO, GRPO, DAPO, etc.) with FSDP, Megatron-LM backends and vLLM, SGLang inference engines.

## Installation & Setup

```bash
# Basic installation
pip install -e .

# With specific dependencies (choose based on needs)
pip install -e .[vllm]     # For vLLM inference backend
pip install -e .[sglang]   # For SGLang inference backend  
pip install -e .[mcore]    # For Megatron-Core support
pip install -e .[gpu]      # For GPU-specific optimizations (flash-attn, liger-kernel)
pip install -e .[test]     # For development and testing
```

## Common Development Commands

### Testing
```bash
# Run unit tests on CPU
pytest tests/ -k "on_cpu"

# Run GPU unit tests (requires GPU)
pytest tests/ -k "not on_cpu"

# Run specific test categories
pytest tests/trainer/         # Training-related tests
pytest tests/models/           # Model-related tests
pytest tests/special_e2e/      # End-to-end tests
```

### Code Quality
```bash
# Run linting (ruff is configured in pyproject.toml)
ruff check .
ruff format .

# Run pre-commit hooks
pre-commit run --all-files

# Type checking (mypy configured in pyproject.toml)
mypy verl/
```

## Training Examples

### PPO Training
```bash
# Basic PPO with reward model
bash examples/ppo_trainer/run_qwen2-7b_rm.sh

# PPO with function-based reward (math)
bash examples/ppo_trainer/run_qwen2-7b_math_gsm8k_megatron.sh
```

### GRPO Training
```bash
# GRPO for math reasoning
bash examples/grpo_trainer/run_qwen2-7b_math.sh

# GRPO with sequence balancing
bash examples/grpo_trainer/run_qwen2-7b_seq_balance.sh
```

### Data Preprocessing
```bash
# Prepare GSM8K dataset
python examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

# Prepare MATH dataset
python examples/data_preprocess/math_dataset.py --local_dir ~/data/math
```

## Architecture

### Core Components

- **`verl/trainer/`**: Training logic and algorithms (PPO, GRPO, etc.)
- **`verl/workers/`**: Distributed worker implementations (actor, critic, rollout)
- **`verl/models/`**: Model implementations for different backends (FSDP, Megatron)
- **`verl/utils/`**: Utilities for checkpointing, datasets, profiling
- **`verl/single_controller/`**: Ray-based distributed coordination

### Configuration System

Uses Hydra for hierarchical configuration:
- Base configs in `verl/trainer/config/`
- Override parameters via command line: `param.subparam=value`
- Example: `actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct`

### Backends & Engines

**Training Backends:**
- FSDP (PyTorch Fully Sharded Data Parallel) - default
- FSDP2 - newer, recommended for better performance
- Megatron-LM - for very large models (DeepSeek-671B, etc.)

**Inference Engines:**
- vLLM - high-performance inference (recommended for production)
- SGLang - multi-turn conversations, tool usage
- HuggingFace Transformers - basic inference

### Multi-GPU Training

Training scripts automatically detect GPU count:
```bash
# Single node, 8 GPUs
trainer.n_gpus_per_node=8 trainer.nnodes=1

# Multi-node with Ray
trainer.n_gpus_per_node=8 trainer.nnodes=4
```

## Key Files

- **`verl/trainer/main_ppo.py`**: Main PPO training entry point
- **`verl/protocol.py`**: Core data structures and protocols
- **`verl/trainer/config/ppo_trainer.yaml`**: Default PPO configuration
- **`setup.py`**: Package dependencies and extras
- **`requirements.txt`**: Development dependencies

## Algorithm-Specific Recipes

Advanced algorithms have dedicated recipes in `recipe/`:
- **`recipe/dapo/`**: DAPO (Direct Advantage Policy Optimization)
- **`recipe/prime/`**: PRIME (Process Reinforcement)  
- **`recipe/sppo/`**: Self-Play Preference Optimization
- **`recipe/r1/`**: R1-style reasoning training

## Development Tips

1. **Config Override Pattern**: Use Hydra's dot notation to override nested configs
2. **Memory Management**: Enable offloading for large models: `fsdp_config.param_offload=True`
3. **Sequence Length**: Use `max_prompt_length` and `max_response_length` to control memory usage
4. **Batch Sizing**: Tune `ppo_micro_batch_size_per_gpu` based on GPU memory
5. **Reward Functions**: Implement custom rewards in `verl/utils/reward_score/`