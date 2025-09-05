# Copyright 2025 User Implementation
# Dual Reward Manager for verl framework

import torch
import warnings
from typing import Any, Dict
from collections import defaultdict
from transformers import (
    AutoConfig, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForSequenceClassification
)
from torch.distributed.fsdp import FSDP, CPUOffload

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.utils.fs import copy_to_local
from verl.utils.tokenizer import hf_tokenizer


@register("dual")
class DualRewardManager(AbstractRewardManager):
    """
    A dual reward manager that manages two independent reward models:
    1. Anthropomorphism scoring model (话术拟人打分模型)
    2. Correctness scoring model (回复正确性模型)
    
    Args:
        tokenizer: The tokenizer for decoding responses
        num_examine: Number of responses to examine for logging
        anthropomorphism_model_path: Path to anthropomorphism scoring model
        correctness_model_path: Path to correctness scoring model
        weight_anthropomorphism: Weight for anthropomorphism score (default: 0.5)
        weight_correctness: Weight for correctness score (default: 0.5)
        device: Device to load models on
        use_trust_remote_code: Whether to trust remote code when loading models
    """
    
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        anthropomorphism_model_path: str,
        correctness_model_path: str,
        weight_anthropomorphism: float = 0.5,
        weight_correctness: float = 0.5,
        device: str = "cuda",
        use_trust_remote_code: bool = False,
        compute_score=None,  # for compatibility with base class
        reward_fn_key: str = "data_source",
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        
        # Weights for combining scores
        assert abs(weight_anthropomorphism + weight_correctness - 1.0) < 1e-6, \
            f"Weights must sum to 1.0, got {weight_anthropomorphism} + {weight_correctness}"
        self.weight_anthropomorphism = weight_anthropomorphism
        self.weight_correctness = weight_correctness
        
        # Device and model config
        self.device = device
        self.use_trust_remote_code = use_trust_remote_code
        
        # Load both models
        self.anthropomorphism_model = self._load_reward_model(anthropomorphism_model_path, "anthropomorphism")
        self.correctness_model = self._load_reward_model(correctness_model_path, "correctness")
        
        print(f"✅ DualRewardManager initialized:")
        print(f"   - Anthropomorphism model: {anthropomorphism_model_path} (weight: {weight_anthropomorphism})")
        print(f"   - Correctness model: {correctness_model_path} (weight: {weight_correctness})")
    
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
    
    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | Dict[str, Any]:
        """
        Compute dual rewards for the given data
        
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
        
        # Extract data
        prompt_ids = data.batch["prompts"]  # [batch_size, prompt_len]
        response_ids = data.batch["responses"]  # [batch_size, response_len]
        attention_mask = data.batch["attention_mask"]  # [batch_size, total_len]
        
        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[-1]
        
        # Construct full input_ids (prompts + responses)
        input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
        
        # Score with both models
        anthropomorphism_scores = self._score_with_model(
            self.anthropomorphism_model, input_ids, attention_mask, "anthropomorphism"
        )
        correctness_scores = self._score_with_model(
            self.correctness_model, input_ids, attention_mask, "correctness"
        )
        
        # Combine scores with weights
        combined_scores = (
            self.weight_anthropomorphism * anthropomorphism_scores +
            self.weight_correctness * correctness_scores
        )
        
        # Create reward tensor with rewards at EOS positions
        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        
        for i in range(batch_size):
            length = valid_response_lengths[i].item()
            if length > 0:
                reward_tensor[i, length - 1] = combined_scores[i]
        
        # Store accuracy info
        data.batch["acc"] = combined_scores.to(prompt_ids.device)
        
        # Print examples for debugging
        self._print_examples(data, anthropomorphism_scores, correctness_scores, combined_scores)
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "anthropomorphism_scores": anthropomorphism_scores,
                "correctness_scores": correctness_scores,
                "combined_scores": combined_scores,
                "reward_extra_info": {
                    "anthropomorphism_scores": anthropomorphism_scores.tolist(),
                    "correctness_scores": correctness_scores.tolist(),
                    "weights": {
                        "anthropomorphism": self.weight_anthropomorphism,
                        "correctness": self.weight_correctness
                    }
                }
            }
        else:
            return reward_tensor
    
    def _print_examples(self, data, anthro_scores, correct_scores, combined_scores):
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
                print("="*80)
                print(f"[DATA_SOURCE] {data_source}")
                print(f"[PROMPT] {prompt_str}")
                print(f"[RESPONSE] {response_str}")
                print(f"[GROUND_TRUTH] {ground_truth}")
                print(f"[ANTHROPOMORPHISM] {anthro_scores[i]:.4f}")
                print(f"[CORRECTNESS] {correct_scores[i]:.4f}")
                print(f"[COMBINED] {combined_scores[i]:.4f} (weights: {self.weight_anthropomorphism:.2f}, {self.weight_correctness:.2f})")
                print("="*80)
                
                already_printed[data_source] = already_printed.get(data_source, 0) + 1