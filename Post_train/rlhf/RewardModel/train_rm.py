#!/usr/bin/env python3
"""
Complete Reward Model training script using Hugging Face official methods.
Based on Qwen 0.6B model with preference data training.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from bitsandbytes.optim import AdamW8bit
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

# Import our custom modules
from reward_model_utils import (
    RewardModel,
    PreferenceDataProcessor,
    RewardModelEvaluator,
    load_preference_dataset,
    save_reward_model,
    load_reward_model
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """Dataset for preference data training."""
    
    def __init__(
        self, 
        data: List[Dict[str, Any]], 
        tokenizer, 
        max_length: int = 512
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get chosen and rejected responses
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        prompt = item.get("prompt", "")
        
        # Tokenize chosen response
        chosen_text = prompt + chosen
        chosen_encoding = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize rejected response
        rejected_text = prompt + rejected
        rejected_encoding = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(0)
        }


class RewardModelTrainer:
    """Complete trainer for Reward Model using preference data."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup quantization config for 8-bit model loading
        quantization_config = None
        if getattr(config, 'use_8bit_model', False):
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            logger.info("Using 8-bit model quantization")
        
        self.model = RewardModel(
            config.model_name, 
            num_labels=1,
            freeze_layers=config.freeze_layers,
            quantization_config=quantization_config
        )
        
        # Enable gradient checkpointing if requested
        if getattr(config, 'gradient_checkpointing', False):
            if hasattr(self.model.base_model, 'gradient_checkpointing_enable'):
                self.model.base_model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                logger.warning("Gradient checkpointing not supported for this model")
        
        # Enable bf16 precision if requested
        if getattr(config, 'use_bf16', False):
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.model = self.model.to(dtype=torch.bfloat16)
                logger.info("Using bf16 precision")
            else:
                logger.warning("bf16 not supported on this device, using fp32")
        
        self.model.to(self.device)
        
        # Initialize data processor
        self.data_processor = PreferenceDataProcessor(
            self.tokenizer, 
            config.max_length
        )
        
        # Initialize evaluator
        self.evaluator = RewardModelEvaluator(
            self.model, 
            self.tokenizer, 
            self.device
        )
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=config.__dict__
            )
    
    def compute_reward_loss(self, chosen_rewards, rejected_rewards, loss_type="ranking"):
        """
        Compute reward model loss with different strategies.
        
        Args:
            chosen_rewards: Reward scores for chosen responses
            rejected_rewards: Reward scores for rejected responses  
            loss_type: Type of loss to compute ("ranking", "mse", "bce")
        
        Returns:
            Dictionary containing loss components
        """
        if loss_type == "ranking":
            # Pairwise ranking loss (recommended for RM)
            # L_RM(φ; x) = -log σ(f_φ(x, y_chosen) - f_φ(x, y_rejected))
            reward_diff = chosen_rewards - rejected_rewards
            ranking_loss = -F.logsigmoid(reward_diff).mean()
            
            # Regularization terms
            chosen_penalty = F.relu(-chosen_rewards).mean()  # Penalty if chosen < 0
            rejected_penalty = F.relu(rejected_rewards).mean()  # Penalty if rejected > 0
            l2_penalty = 0.01 * (chosen_rewards.pow(2).mean() + rejected_rewards.pow(2).mean())
            
            total_loss = ranking_loss + 0.1 * (chosen_penalty + rejected_penalty) + l2_penalty
            
            return {
                "loss": total_loss,
                "ranking_loss": ranking_loss,
                "chosen_penalty": chosen_penalty,
                "rejected_penalty": rejected_penalty,
                "l2_penalty": l2_penalty,
                "reward_diff": reward_diff.mean()
            }
            
        elif loss_type == "mse":
            # MSE loss (not recommended for RM, but kept for comparison)
            chosen_target = torch.ones_like(chosen_rewards)
            rejected_target = torch.zeros_like(rejected_rewards)
            
            chosen_loss = F.mse_loss(chosen_rewards, chosen_target)
            rejected_loss = F.mse_loss(rejected_rewards, rejected_target)
            total_loss = chosen_loss + rejected_loss
            
            return {
                "loss": total_loss,
                "chosen_loss": chosen_loss,
                "rejected_loss": rejected_loss,
                "reward_diff": (chosen_rewards - rejected_rewards).mean()
            }
            
        elif loss_type == "bce":
            # Binary cross-entropy loss
            chosen_target = torch.ones_like(chosen_rewards)
            rejected_target = torch.zeros_like(rejected_rewards)
            
            chosen_loss = F.binary_cross_entropy_with_logits(chosen_rewards, chosen_target)
            rejected_loss = F.binary_cross_entropy_with_logits(rejected_rewards, rejected_target)
            total_loss = chosen_loss + rejected_loss
            
            return {
                "loss": total_loss,
                "chosen_loss": chosen_loss,
                "rejected_loss": rejected_loss,
                "reward_diff": (chosen_rewards - rejected_rewards).mean()
            }
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load preference data from Hugging Face datasets."""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        # Load training data
        train_data = load_preference_dataset(
            self.config.dataset_name, 
            self.config.train_split
        )
        
        # Load evaluation data
        eval_data = load_preference_dataset(
            self.config.dataset_name, 
            self.config.eval_split
        )
        
        # Limit data size for faster training (optional)
        if self.config.max_train_samples > 0:
            train_data = train_data[:self.config.max_train_samples]
        if self.config.max_eval_samples > 0:
            eval_data = eval_data[:self.config.max_eval_samples]
        
        logger.info(f"Loaded {len(train_data)} training samples, {len(eval_data)} eval samples")
        return train_data, eval_data
    
    def create_datasets(self, train_data: List[Dict], eval_data: List[Dict]):
        """Create PyTorch datasets."""
        train_dataset = PreferenceDataset(
            train_data, 
            self.tokenizer, 
            self.config.max_length
        )
        eval_dataset = PreferenceDataset(
            eval_data, 
            self.tokenizer, 
            self.config.max_length
        )
        return train_dataset, eval_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        # For reward model, we compute regression metrics
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": np.sqrt(mse)
        }
    
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        
        # Get chosen and rejected data
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
        
        # Forward pass for chosen responses
        chosen_output = self.model(chosen_input_ids, chosen_attention_mask)
        chosen_rewards = chosen_output["rewards"]
        
        # Forward pass for rejected responses
        rejected_output = self.model(rejected_input_ids, rejected_attention_mask)
        rejected_rewards = rejected_output["rewards"]
        
        # Compute loss using the new loss function
        # Default to ranking loss (recommended for RM)
        loss_type = getattr(self.config, 'loss_type', 'ranking')
        loss_metrics = self.compute_reward_loss(chosen_rewards, rejected_rewards, loss_type)
        
        # Add additional metrics
        loss_metrics.update({
            "chosen_rewards": chosen_rewards.mean(),
            "rejected_rewards": rejected_rewards.mean()
        })
        
        return loss_metrics
    
    def train(self, train_data: List[Dict], eval_data: List[Dict]):
        """Train the reward model."""
        logger.info("Starting reward model training...")
        
        # Create datasets
        train_dataset, eval_dataset = self.create_datasets(train_data, eval_data)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        # Optimizer - use 8-bit AdamW if requested
        if getattr(self.config, 'use_8bit_optimizer', False):
            optimizer = AdamW8bit(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            logger.info("Using 8-bit AdamW optimizer")
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            logger.info("Using standard AdamW optimizer")
        
        # Log memory optimization status
        opt_8bit = getattr(self.config, 'use_8bit_optimizer', False)
        model_8bit = getattr(self.config, 'use_8bit_model', False)
        grad_checkpoint = getattr(self.config, 'gradient_checkpointing', False)
        bf16_enabled = getattr(self.config, 'use_bf16', False)
        
        logger.info(f"Memory optimization status:")
        logger.info(f"  - 8-bit optimizer: {opt_8bit}")
        logger.info(f"  - 8-bit model: {model_8bit}")
        logger.info(f"  - Gradient checkpointing: {grad_checkpoint}")
        logger.info(f"  - bf16 precision: {bf16_enabled}")
        
        # Learning rate scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps
        )
        
        # Training loop
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            self.model.train()
            train_loss = 0
            
            # Initialize metrics based on loss type
            if self.config.loss_type == "ranking":
                train_ranking_loss = 0
                train_chosen_penalty = 0
                train_rejected_penalty = 0
                train_l2_penalty = 0
            elif self.config.loss_type in ["mse", "bce"]:
                train_chosen_loss = 0
                train_rejected_loss = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
                # Training step
                step_metrics = self.train_step(batch)
                
                # Backward pass
                optimizer.zero_grad()
                step_metrics["loss"].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                optimizer.step()
                scheduler.step()
                
                # Update metrics based on loss type
                train_loss += step_metrics["loss"].item()
                
                # Handle different loss types
                if self.config.loss_type == "ranking":
                    train_ranking_loss += step_metrics["ranking_loss"].item()
                    train_chosen_penalty += step_metrics["chosen_penalty"].item()
                    train_rejected_penalty += step_metrics["rejected_penalty"].item()
                    train_l2_penalty += step_metrics["l2_penalty"].item()
                elif self.config.loss_type in ["mse", "bce"]:
                    train_chosen_loss += step_metrics["chosen_loss"].item()
                    train_rejected_loss += step_metrics["rejected_loss"].item()
                
                global_step += 1
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = train_loss / (batch_idx + 1)
                    
                    if self.config.loss_type == "ranking":
                        avg_ranking_loss = train_ranking_loss / (batch_idx + 1)
                        avg_chosen_penalty = train_chosen_penalty / (batch_idx + 1)
                        avg_rejected_penalty = train_rejected_penalty / (batch_idx + 1)
                        avg_l2_penalty = train_l2_penalty / (batch_idx + 1)
                        
                        logger.info(
                            f"Step {global_step}: "
                            f"Loss: {avg_loss:.4f}, "
                            f"Ranking Loss: {avg_ranking_loss:.4f}, "
                            f"Chosen Penalty: {avg_chosen_penalty:.4f}, "
                            f"Rejected Penalty: {avg_rejected_penalty:.4f}, "
                            f"L2 Penalty: {avg_l2_penalty:.4f}"
                        )
                        
                        if self.config.use_wandb:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/ranking_loss": avg_ranking_loss,
                                "train/chosen_penalty": avg_chosen_penalty,
                                "train/rejected_penalty": avg_rejected_penalty,
                                "train/l2_penalty": avg_l2_penalty,
                                "train/learning_rate": scheduler.get_last_lr()[0],
                                "train/step": global_step
                            })
                    
                    elif self.config.loss_type in ["mse", "bce"]:
                        avg_chosen_loss = train_chosen_loss / (batch_idx + 1)
                        avg_rejected_loss = train_rejected_loss / (batch_idx + 1)
                        
                        logger.info(
                            f"Step {global_step}: "
                            f"Loss: {avg_loss:.4f}, "
                            f"Chosen Loss: {avg_chosen_loss:.4f}, "
                            f"Rejected Loss: {avg_rejected_loss:.4f}"
                        )
                        
                        if self.config.use_wandb:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/chosen_loss": avg_chosen_loss,
                                "train/rejected_loss": avg_rejected_loss,
                                "train/learning_rate": scheduler.get_last_lr()[0],
                                "train/step": global_step
                            })
                
                # Evaluation
                if global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(eval_loader)
                    
                    if self.config.loss_type == "ranking":
                        logger.info(
                            f"Evaluation at step {global_step}: "
                            f"Loss: {eval_metrics['loss']:.4f}, "
                            f"Ranking Loss: {eval_metrics['ranking_loss']:.4f}, "
                            f"Chosen Penalty: {eval_metrics['chosen_penalty']:.4f}, "
                            f"Rejected Penalty: {eval_metrics['rejected_penalty']:.4f}, "
                            f"L2 Penalty: {eval_metrics['l2_penalty']:.4f}, "
                            f"Reward Diff: {eval_metrics['reward_diff']:.4f}"
                        )
                        
                        if self.config.use_wandb:
                            wandb.log({
                                "eval/loss": eval_metrics["loss"],
                                "eval/ranking_loss": eval_metrics["ranking_loss"],
                                "eval/chosen_penalty": eval_metrics["chosen_penalty"],
                                "eval/rejected_penalty": eval_metrics["rejected_penalty"],
                                "eval/l2_penalty": eval_metrics["l2_penalty"],
                                "eval/reward_diff": eval_metrics["reward_diff"],
                                "eval/step": global_step
                            })
                        
                        # Save best model based on ranking loss
                        if eval_metrics["ranking_loss"] < best_eval_loss:
                            best_eval_loss = eval_metrics["ranking_loss"]
                            self.save_model(f"best_model_step_{global_step}")
                    
                    elif self.config.loss_type in ["mse", "bce"]:
                        logger.info(
                            f"Evaluation at step {global_step}: "
                            f"Loss: {eval_metrics['loss']:.4f}, "
                            f"Chosen Loss: {eval_metrics['chosen_loss']:.4f}, "
                            f"Rejected Loss: {eval_metrics['rejected_loss']:.4f}, "
                            f"Reward Diff: {eval_metrics['reward_diff']:.4f}"
                        )
                        
                        if self.config.use_wandb:
                            wandb.log({
                                "eval/loss": eval_metrics["loss"],
                                "eval/chosen_loss": eval_metrics["chosen_loss"],
                                "eval/rejected_loss": eval_metrics["rejected_loss"],
                                "eval/reward_diff": eval_metrics["reward_diff"],
                                "eval/step": global_step
                            })
                        
                        # Save best model based on total loss
                        if eval_metrics["loss"] < best_eval_loss:
                            best_eval_loss = eval_metrics["loss"]
                            self.save_model(f"best_model_step_{global_step}")
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_model(f"checkpoint_step_{global_step}")
        
        # Final evaluation
        logger.info("Final evaluation...")
        final_eval_metrics = self.evaluate(eval_loader)
        logger.info(f"Final evaluation metrics: {final_eval_metrics}")
        
        # Save final model
        self.save_model("final_model")
        
        logger.info(f"Training completed. Model saved to {self.config.output_dir}")
        
        return final_eval_metrics
    
    def evaluate(self, eval_loader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total_reward_diff = 0
        total_chosen_rewards = 0
        total_rejected_rewards = 0
        num_batches = 0
        
        # Initialize metrics based on loss type
        if self.config.loss_type == "ranking":
            total_ranking_loss = 0
            total_chosen_penalty = 0
            total_rejected_penalty = 0
            total_l2_penalty = 0
        elif self.config.loss_type in ["mse", "bce"]:
            total_chosen_loss = 0
            total_rejected_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                step_metrics = self.train_step(batch)
                
                total_loss += step_metrics["loss"].item()
                total_reward_diff += step_metrics["reward_diff"].item()
                total_chosen_rewards += step_metrics["chosen_rewards"].item()
                total_rejected_rewards += step_metrics["rejected_rewards"].item()
                
                # Update metrics based on loss type
                if self.config.loss_type == "ranking":
                    total_ranking_loss += step_metrics["ranking_loss"].item()
                    total_chosen_penalty += step_metrics["chosen_penalty"].item()
                    total_rejected_penalty += step_metrics["rejected_penalty"].item()
                    total_l2_penalty += step_metrics["l2_penalty"].item()
                elif self.config.loss_type in ["mse", "bce"]:
                    total_chosen_loss += step_metrics["chosen_loss"].item()
                    total_rejected_loss += step_metrics["rejected_loss"].item()
                
                num_batches += 1
        
        # Return metrics based on loss type
        base_metrics = {
            "loss": total_loss / num_batches,
            "reward_diff": total_reward_diff / num_batches,
            "chosen_rewards": total_chosen_rewards / num_batches,
            "rejected_rewards": total_rejected_rewards / num_batches
        }
        
        if self.config.loss_type == "ranking":
            base_metrics.update({
                "ranking_loss": total_ranking_loss / num_batches,
                "chosen_penalty": total_chosen_penalty / num_batches,
                "rejected_penalty": total_rejected_penalty / num_batches,
                "l2_penalty": total_l2_penalty / num_batches
            })
        elif self.config.loss_type in ["mse", "bce"]:
            base_metrics.update({
                "chosen_loss": total_chosen_loss / num_batches,
                "rejected_loss": total_rejected_loss / num_batches
            })
        
        return base_metrics
    
    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        return {
            "chosen_input_ids": torch.stack([item["chosen_input_ids"] for item in batch]),
            "chosen_attention_mask": torch.stack([item["chosen_attention_mask"] for item in batch]),
            "rejected_input_ids": torch.stack([item["rejected_input_ids"] for item in batch]),
            "rejected_attention_mask": torch.stack([item["rejected_attention_mask"] for item in batch])
        }
    
    def save_model(self, name: str):
        """Save the model."""
        save_dir = os.path.join(self.config.output_dir, name)
        save_reward_model(self.model, self.tokenizer, save_dir)
        logger.info(f"Model saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Reward Model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--output_dir", type=str, default="./reward_model_checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--freeze_layers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=10000)
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="reward-model-training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    
    # Memory optimization
    parser.add_argument("--use_8bit_optimizer", action="store_true", 
                       help="Use 8-bit AdamW optimizer (saves optimizer memory)")
    parser.add_argument("--use_8bit_model", action="store_true",
                       help="Use 8-bit model quantization (saves model memory)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing to save memory")
    parser.add_argument("--use_bf16", action="store_true",
                       help="Use bf16 precision for training (saves memory, better stability)")
    
    args = parser.parse_args()
    
    # Create config
    config = argparse.Namespace(**vars(args))
    
    # Create trainer
    trainer = RewardModelTrainer(config)
    
    # Load data
    train_data, eval_data = trainer.load_data()
    
    # Train model
    trainer.train(train_data, eval_data)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
