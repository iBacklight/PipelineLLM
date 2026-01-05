#!/usr/bin/env python3
"""
Utility functions for Reward Model training and evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json
import logging
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """Reward Model based on Qwen architecture for preference learning."""
    
    def __init__(
        self, 
        model_name: str, 
        num_labels: int = 1,
        freeze_layers: int = 0,
        hidden_size: Optional[int] = None,
        quantization_config: Optional[Any] = None
    ):
        super().__init__()
        
        # Load the base model with optional quantization
        if quantization_config is not None:
            self.base_model = AutoModel.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            self.base_model = AutoModel.from_pretrained(model_name)
        self.config = self.base_model.config
        
        # Get hidden size
        if hidden_size is None:
            hidden_size = self.config.hidden_size
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Freeze layers if specified
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
    
    def _freeze_layers(self, freeze_layers: int):
        """Freeze the first few layers of the model."""
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, 'layers'):
            layers = self.base_model.layers
        else:
            logger.warning("Could not find layers to freeze")
            return
        
        for i, layer in enumerate(layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                logger.info(f"Frozen layer {i}")
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the reward model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            labels: Optional labels for training
        
        Returns:
            Dictionary containing rewards and loss (if labels provided)
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use the last hidden state (CLS token or mean pooling)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Pool over sequence length (mean pooling)
        pooled_output = hidden_states.mean(dim=1)  # (batch_size, hidden_size)
        
        # Compute rewards
        rewards = self.reward_head(pooled_output).squeeze(-1)  # (batch_size,)
        
        result = {"rewards": rewards}
        
        # Compute loss if labels provided
        if labels is not None:
            loss = F.mse_loss(rewards, labels)
            result["loss"] = loss
        
        return result


class PreferenceDataProcessor:
    """Process preference data for reward model training."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def process_preference_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process preference data into training format.
        
        Args:
            data: List of preference data items
        
        Returns:
            Processed data ready for training
        """
        processed_data = []
        
        for item in tqdm(data, desc="Processing preference data"):
            # Extract prompt and responses
            prompt = item.get("prompt", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            
            if not chosen or not rejected:
                continue
            
            # Create training examples
            # Example 1: chosen response (positive)
            chosen_text = prompt + chosen
            chosen_tokens = self.tokenizer(
                chosen_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Example 2: rejected response (negative)
            rejected_text = prompt + rejected
            rejected_tokens = self.tokenizer(
                rejected_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            processed_data.append({
                "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
                "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
                "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
                "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
                "chosen_reward": 1.0,  # Positive reward
                "rejected_reward": 0.0,  # Negative reward
            })
        
        return processed_data
    
    def create_ranking_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create ranking data for pairwise training.
        
        Args:
            data: List of preference data items
        
        Returns:
            Ranking data for pairwise training
        """
        ranking_data = []
        
        for item in tqdm(data, desc="Creating ranking data"):
            prompt = item.get("prompt", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            
            if not chosen or not rejected:
                continue
            
            # Create pairwise comparison
            chosen_text = prompt + chosen
            rejected_text = prompt + rejected
            
            # Tokenize both
            chosen_tokens = self.tokenizer(
                chosen_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            rejected_tokens = self.tokenizer(
                rejected_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            ranking_data.append({
                "input_ids": torch.cat([
                    chosen_tokens["input_ids"],
                    rejected_tokens["input_ids"]
                ], dim=0),
                "attention_mask": torch.cat([
                    chosen_tokens["attention_mask"],
                    rejected_tokens["attention_mask"]
                ], dim=0),
                "labels": torch.tensor([1.0, 0.0]),  # chosen > rejected
                "is_chosen": torch.tensor([True, False])
            })
        
        return ranking_data


class RewardModelEvaluator:
    """Evaluator for reward model performance."""
    
    def __init__(self, model: RewardModel, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def evaluate_preference_accuracy(
        self, 
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate preference prediction accuracy.
        
        Args:
            test_data: Test preference data
        
        Returns:
            Evaluation metrics
        """
        correct = 0
        total = 0
        rewards_chosen = []
        rewards_rejected = []
        
        with torch.no_grad():
            for item in tqdm(test_data, desc="Evaluating preference accuracy"):
                prompt = item.get("prompt", "")
                chosen = item.get("chosen", "")
                rejected = item.get("rejected", "")
                
                if not chosen or not rejected:
                    continue
                
                # Get rewards for both responses
                chosen_text = prompt + chosen
                rejected_text = prompt + rejected
                
                chosen_tokens = self.tokenizer(
                    chosen_text,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)
                
                rejected_tokens = self.tokenizer(
                    rejected_text,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)
                
                # Get rewards
                chosen_output = self.model(
                    chosen_tokens["input_ids"],
                    chosen_tokens["attention_mask"]
                )
                rejected_output = self.model(
                    rejected_tokens["input_ids"],
                    rejected_tokens["attention_mask"]
                )
                
                chosen_reward = chosen_output["rewards"].item()
                rejected_reward = rejected_output["rewards"].item()
                
                rewards_chosen.append(chosen_reward)
                rewards_rejected.append(rejected_reward)
                
                # Check if preference is correct
                if chosen_reward > rejected_reward:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "total_samples": total,
            "correct_predictions": correct,
            "mean_chosen_reward": np.mean(rewards_chosen),
            "mean_rejected_reward": np.mean(rewards_rejected),
            "reward_difference": np.mean(rewards_chosen) - np.mean(rewards_rejected)
        }
    
    def evaluate_reward_distribution(
        self, 
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate reward distribution statistics.
        
        Args:
            test_data: Test data
        
        Returns:
            Reward distribution metrics
        """
        all_rewards = []
        
        with torch.no_grad():
            for item in tqdm(test_data, desc="Evaluating reward distribution"):
                prompt = item.get("prompt", "")
                chosen = item.get("chosen", "")
                
                if not chosen:
                    continue
                
                text = prompt + chosen
                tokens = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)
                
                output = self.model(
                    tokens["input_ids"],
                    tokens["attention_mask"]
                )
                
                all_rewards.append(output["rewards"].item())
        
        all_rewards = np.array(all_rewards)
        
        return {
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "min_reward": np.min(all_rewards),
            "max_reward": np.max(all_rewards),
            "median_reward": np.median(all_rewards),
            "q25_reward": np.percentile(all_rewards, 25),
            "q75_reward": np.percentile(all_rewards, 75)
        }


def load_preference_dataset(dataset_name: str, split: str = "train") -> List[Dict[str, Any]]:
    """
    Load preference dataset from Hugging Face.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split to load
    
    Returns:
        List of preference data items
    """
    from datasets import load_dataset
    
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    data = []
    for item in dataset[split]:
        # Handle different dataset formats
        if "chosen" in item and "rejected" in item:
            data.append({
                "prompt": item.get("prompt", ""),
                "chosen": item["chosen"],
                "rejected": item["rejected"]
            })
        elif "human_ref" in item and "gpt_ref" in item:
            data.append({
                "prompt": item.get("prompt", ""),
                "chosen": item["human_ref"],
                "rejected": item["gpt_ref"]
            })
        elif "response_j" in item and "response_k" in item:
            data.append({
                "prompt": item.get("prompt", ""),
                "chosen": item["response_j"],
                "rejected": item["response_k"]
            })
    
    logger.info(f"Loaded {len(data)} samples from {dataset_name}")
    return data


def save_reward_model(model: RewardModel, tokenizer, output_dir: str):
    """Save the trained reward model."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    
    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(model.config.to_dict(), f, indent=2)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Reward model saved to {output_dir}")


def load_reward_model(model_name: str, output_dir: str, device: str = "cuda") -> Tuple[RewardModel, AutoTokenizer]:
    """Load a trained reward model."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    
    # Load model
    model = RewardModel(model_name)
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    return model, tokenizer
