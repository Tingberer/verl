# Copyright 2025 User Implementation
# Hybrid Reward Manager: Rule-based + Dual Reward Models for verl framework

import torch
import warnings
from typing import Any, Dict, Callable, Optional
from collections import defaultdict
from transformers import (
    AutoConfig, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForSequenceClassification
)

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import default_compute_score


@register("hybrid")
class HybridRewardManager(AbstractRewardManager):
    """
    Hybrid reward manager that combines:
    1. Rule-based reward (custom function)
    2. Anthropomorphism scoring model (话术拟人打分模型) 
    3. Correctness scoring model (回复正确性模型)
    
    Final reward = w1 * rule_score + w2 * anthropomorphism_score + w3 * correctness_score
    
    Args:
        tokenizer: The tokenizer for decoding responses
        num_examine: Number of responses to examine for logging
        compute_score: Rule-based reward function (can be None)
        anthropomorphism_model_path: Path to anthropomorphism scoring model
        correctness_model_path: Path to correctness scoring model
        weight_rule: Weight for rule-based score (default: 0.3)
        weight_anthropomorphism: Weight for anthropomorphism score (default: 0.4)
        weight_correctness: Weight for correctness score (default: 0.3)
        device: Device to load models on
        use_trust_remote_code: Whether to trust remote code when loading models
        reward_fn_key: Key for data source in non_tensor_batch
    """
    
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: Optional[RawRewardFn] = None,
        anthropomorphism_model_path: Optional[str] = None,
        correctness_model_path: Optional[str] = None,
        weight_rule: float = 0.3,
        weight_anthropomorphism: float = 0.4,
        weight_correctness: float = 0.3,
        device: str = "cuda",
        use_trust_remote_code: bool = False,
        reward_fn_key: str = "data_source",
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        
        # Validate weights
        total_weight = weight_rule + weight_anthropomorphism + weight_correctness
        assert abs(total_weight - 1.0) < 1e-6, \
            f"Weights must sum to 1.0, got {total_weight} (rule: {weight_rule}, anthro: {weight_anthropomorphism}, correct: {weight_correctness})"
        
        self.weight_rule = weight_rule
        self.weight_anthropomorphism = weight_anthropomorphism
        self.weight_correctness = weight_correctness
        
        # Rule-based reward function
        self.compute_score = compute_score or default_compute_score
        
        # Device and model config
        self.device = device
        self.use_trust_remote_code = use_trust_remote_code
        
        # Load neural reward models (both optional)
        self.anthropomorphism_model = None
        self.correctness_model = None
        
        if anthropomorphism_model_path:
            self.anthropomorphism_model = self._load_reward_model(
                anthropomorphism_model_path, "anthropomorphism"
            )
        
        if correctness_model_path:
            self.correctness_model = self._load_reward_model(
                correctness_model_path, "correctness"
            )
        
        # Validate model configuration
        if self.weight_anthropomorphism > 0 and self.anthropomorphism_model is None:
            raise ValueError("anthropomorphism_model_path required when weight_anthropomorphism > 0")
        if self.weight_correctness > 0 and self.correctness_model is None:
            raise ValueError("correctness_model_path required when weight_correctness > 0")
        
        self._print_initialization_info()
    
    def _load_reward_model(self, model_path: str, model_type: str):
        """Load a single reward model from path with intelligent architecture detection"""
        print(f"🔄 Loading {model_type} reward model from: {model_path}")
        
        # Download from HDFS if needed
        local_path = copy_to_local(model_path, use_shm=False)
        
        # Load model config to detect architecture
        model_config = AutoConfig.from_pretrained(
            local_path, 
            trust_remote_code=self.use_trust_remote_code
        )
        
        # Detect model architecture and load appropriately
        model_info = self._detect_model_architecture(model_config, model_type, local_path)
        
        print(f"📋 {model_type} architecture detected: {model_info['architecture']}")
        print(f"📋 {model_type} model class: {model_info['model_class'].__name__}")
        
        # Load model with detected settings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Apply model-specific configurations
            if model_info['config_updates']:
                for key, value in model_info['config_updates'].items():
                    setattr(model_config, key, value)
            
            model = model_info['model_class'].from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=self.use_trust_remote_code,
                **model_info.get('load_kwargs', {})
            )
            
            # Apply post-load modifications if needed
            if model_info.get('post_load_fn'):
                model = model_info['post_load_fn'](model)
            
            # Move to device and set eval mode
            model.to(self.device)
            model.eval()
            
        print(f"✅ {model_type} model loaded successfully")
        return model
    
    def _detect_model_architecture(self, config, model_type: str, local_path: str) -> dict:
        """Detect model architecture and return appropriate loading configuration"""
        architecture = getattr(config, 'model_type', 'unknown').lower()
        
        # Default configuration
        model_info = {
            'architecture': architecture,
            'model_class': AutoModelForSequenceClassification,
            'config_updates': {'num_labels': 1, 'classifier_dropout': 0.0},
            'load_kwargs': {},
            'post_load_fn': None
        }
        
        # Special handling for different architectures
        if 'qwen' in architecture or 'qwen' in model_type.lower():
            # Qwen-based models (including Qwen3 4B for anthropomorphism)
            print(f"📌 Detected Qwen-based model for {model_type}")
            
            if model_type.lower() == 'anthropomorphism':
                print("🎭 Loading Qwen3 4B-based anthropomorphism model")
                # Qwen3 4B for anthropomorphism scoring
                model_info.update({
                    'architecture': 'qwen3-4b-anthropomorphism',
                    'config_updates': {
                        'num_labels': 1,
                        'classifier_dropout': 0.0,
                        'use_cache': False  # For memory efficiency
                    }
                })
            
        elif 'deepseek' in architecture or 'deepseek' in model_type.lower() or 'deepseek' in local_path.lower():
            # DeepSeek-based models (DeepSeek-R1-Distill-Qwen-14B for correctness)
            print(f"📌 Detected DeepSeek-based model for {model_type}")
            
            if model_type.lower() == 'correctness':
                print("✅ Loading DeepSeek-R1-Distill-Qwen-14B-based correctness model")
                # DeepSeek-R1-Distill-Qwen-14B for correctness scoring
                model_info.update({
                    'architecture': 'deepseek-r1-distill-qwen-14b-correctness',
                    'config_updates': {
                        'num_labels': 1,
                        'classifier_dropout': 0.0,
                        'use_cache': False,
                        'torch_dtype': torch.bfloat16
                    },
                    'load_kwargs': {
                        'device_map': None,  # We handle device placement manually
                        'low_cpu_mem_usage': True
                    }
                })
                
                # Special handling for DeepSeek R1 models if needed
                def post_load_deepseek(model):
                    """Post-processing for DeepSeek models"""
                    # Ensure proper head configuration
                    if hasattr(model, 'classifier') and model.classifier.out_features != 1:
                        print("🔧 Adjusting DeepSeek classifier head for reward scoring")
                        import torch.nn as nn
                        model.classifier = nn.Linear(
                            model.classifier.in_features, 
                            1, 
                            bias=model.classifier.bias is not None
                        ).to(model.classifier.weight.device)
                    return model
                
                model_info['post_load_fn'] = post_load_deepseek
        
        # Fallback for other architectures
        else:
            print(f"⚠️  Unknown architecture '{architecture}' for {model_type}, using default AutoModel")
        
        return model_info
    
    def _print_initialization_info(self):
        """Print initialization information"""
        print("=" * 80)
        print("🎯 HybridRewardManager Initialized")
        print(f"   Rule-based reward: {'✅' if self.compute_score else '❌'} (weight: {self.weight_rule:.3f})")
        print(f"   Anthropomorphism model: {'✅' if self.anthropomorphism_model else '❌'} (weight: {self.weight_anthropomorphism:.3f})")
        print(f"   Correctness model: {'✅' if self.correctness_model else '❌'} (weight: {self.weight_correctness:.3f})")
        print("=" * 80)
    
    def _compute_rule_based_rewards(self, data: DataProto) -> torch.Tensor:
        """Compute rule-based rewards for all items in batch"""
        rule_rewards = []
        
        for i in range(len(data)):
            data_item = data[i]
            
            # Extract and decode text
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            prompt_length = prompt_ids.shape[-1]
            
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # Get metadata
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            
            # Compute rule-based score
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            # Extract score value
            if isinstance(score, dict):
                rule_rewards.append(score["score"])
            else:
                rule_rewards.append(score)
        
        return torch.tensor(rule_rewards, dtype=torch.float32)
    
    def _score_with_model(self, model, input_ids, attention_mask, model_type="unknown"):
        """Score responses using a single reward model with architecture-aware scoring"""
        with torch.no_grad():
            # Move inputs to model device
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            
            try:
                # Forward pass with error handling for different architectures
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Handle different output formats
                if hasattr(outputs, 'logits'):
                    scores = outputs.logits
                elif hasattr(outputs, 'prediction_scores'):
                    scores = outputs.prediction_scores
                elif isinstance(outputs, torch.Tensor):
                    scores = outputs
                else:
                    raise ValueError(f"Unexpected model output format for {model_type}: {type(outputs)}")
                
                # Ensure proper shape for reward scoring
                if scores.dim() > 2:
                    scores = scores.squeeze(-1)  # Remove last dimension if it's 1
                
                # Extract reward at EOS positions
                seq_lengths = attention_mask.sum(dim=-1)  # [batch_size]
                batch_size = input_ids.shape[0]
                
                reward_scores = []
                for i in range(batch_size):
                    eos_pos = seq_lengths[i] - 1  # Last valid position
                    
                    if scores.dim() == 2:  # [batch_size, seq_len]
                        score_value = scores[i, eos_pos].item()
                    elif scores.dim() == 1:  # [batch_size]
                        score_value = scores[i].item()
                    else:
                        # Fallback: take mean of all positions
                        valid_scores = scores[i, :seq_lengths[i]]
                        score_value = valid_scores.mean().item()
                    
                    reward_scores.append(score_value)
                
                return torch.tensor(reward_scores, dtype=torch.float32)
                
            except Exception as e:
                print(f"⚠️  Warning: Error scoring with {model_type} model: {e}")
                print(f"Input shape: {input_ids.shape}, Attention mask shape: {attention_mask.shape}")
                # Return zero scores as fallback
                return torch.zeros(input_ids.shape[0], dtype=torch.float32)
    
    def _compute_neural_rewards(self, data: DataProto) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute neural model rewards (anthropomorphism and correctness)"""
        # Extract data
        prompt_ids = data.batch["prompts"]  # [batch_size, prompt_len]
        response_ids = data.batch["responses"]  # [batch_size, response_len]
        attention_mask = data.batch["attention_mask"]  # [batch_size, total_len]
        
        # Construct full input_ids (prompts + responses)
        input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
        
        # Score with models
        anthropomorphism_scores = torch.zeros(len(data), dtype=torch.float32)
        correctness_scores = torch.zeros(len(data), dtype=torch.float32)
        
        if self.anthropomorphism_model:
            anthropomorphism_scores = self._score_with_model(
                self.anthropomorphism_model, input_ids, attention_mask, "anthropomorphism"
            )
        
        if self.correctness_model:
            correctness_scores = self._score_with_model(
                self.correctness_model, input_ids, attention_mask, "correctness"
            )
        
        return anthropomorphism_scores, correctness_scores
    
    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | Dict[str, Any]:
        """
        Compute hybrid rewards combining rule-based and neural model rewards
        
        Args:
            data: DataProto containing prompts, responses, attention_mask
            return_dict: Whether to return detailed results dict
            
        Returns:
            Combined reward tensor or detailed dict with individual scores
        """
        # If pre-computed rewards exist, return them
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        
        # Compute all three types of rewards
        rule_scores = self._compute_rule_based_rewards(data) if self.weight_rule > 0 else torch.zeros(len(data))
        anthropomorphism_scores, correctness_scores = self._compute_neural_rewards(data)
        
        # Combine scores with weights
        combined_scores = (
            self.weight_rule * rule_scores +
            self.weight_anthropomorphism * anthropomorphism_scores +
            self.weight_correctness * correctness_scores
        )
        
        # Create reward tensor with rewards at EOS positions
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        prompt_len = prompt_ids.shape[-1]
        
        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        
        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            if length > 0:
                reward_tensor[i, length - 1] = combined_scores[i]
        
        # Store accuracy info
        data.batch["acc"] = combined_scores.to(prompt_ids.device)
        
        # Print examples for debugging
        self._print_examples(data, rule_scores, anthropomorphism_scores, correctness_scores, combined_scores)
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "rule_scores": rule_scores,
                "anthropomorphism_scores": anthropomorphism_scores,
                "correctness_scores": correctness_scores,
                "combined_scores": combined_scores,
                "reward_extra_info": {
                    "rule_scores": rule_scores.tolist(),
                    "anthropomorphism_scores": anthropomorphism_scores.tolist(),
                    "correctness_scores": correctness_scores.tolist(),
                    "weights": {
                        "rule": self.weight_rule,
                        "anthropomorphism": self.weight_anthropomorphism,
                        "correctness": self.weight_correctness
                    }
                }
            }
        else:
            return reward_tensor
    
    def _print_examples(self, data, rule_scores, anthro_scores, correct_scores, combined_scores):
        """Print example responses for debugging"""
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        prompt_len = prompt_ids.shape[-1]
        
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        
        already_printed = {}
        
        for i in range(min(len(data), self.num_examine)):
            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                length = valid_response_lengths[i].item()
                
                # Decode texts
                prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                response_str = self.tokenizer.decode(response_ids[i][:length], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                
                # Print detailed info
                print("=" * 100)
                print(f"[DATA_SOURCE] {data_source}")
                print(f"[PROMPT] {prompt_str}")
                print(f"[RESPONSE] {response_str}")
                print(f"[GROUND_TRUTH] {ground_truth}")
                print("-" * 50)
                print(f"[RULE_BASED] {rule_scores[i]:.4f} (weight: {self.weight_rule:.3f})")
                print(f"[ANTHROPOMORPHISM] {anthro_scores[i]:.4f} (weight: {self.weight_anthropomorphism:.3f})")
                print(f"[CORRECTNESS] {correct_scores[i]:.4f} (weight: {self.weight_correctness:.3f})")
                print(f"[COMBINED] {combined_scores[i]:.4f}")
                print("-" * 50)
                contribution_rule = self.weight_rule * rule_scores[i]
                contribution_anthro = self.weight_anthropomorphism * anthro_scores[i]
                contribution_correct = self.weight_correctness * correct_scores[i]
                print(f"[CONTRIBUTIONS] Rule: {contribution_rule:.4f}, Anthro: {contribution_anthro:.4f}, Correct: {contribution_correct:.4f}")
                print("=" * 100)
                
                already_printed[data_source] = already_printed.get(data_source, 0) + 1