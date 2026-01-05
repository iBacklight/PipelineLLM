#!/usr/bin/env python3
"""
DPO Training Script using TRL and Unsloth
=========================================

This script implements DPO training using TRL's DPOTrainer and Unsloth for efficient training.
Uses the unsloth/Qwen3-1.7B-unsloth-bnb-4bit model with 4-bit quantization.

Features:
- TRL DPOTrainer for robust DPO implementation
- Unsloth for memory-efficient training
- 4-bit quantization support
- Wandb logging
- Configurable training parameters

Usage:
    python train_dpo_trl.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Import unsloth first for optimizations
from unsloth import FastLanguageModel

import torch
import wandb
from datasets import load_from_disk, Dataset
from transformers import BitsAndBytesConfig
from trl import DPOTrainer
from unsloth_compiled_cache.UnslothDPOTrainer import UnslothDPOConfig
from unsloth import is_bfloat16_supported
import torch.nn.functional as F

# Add project root to path
current_path = Path(__file__).resolve()
project_root = current_path.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# =========================
# Configuration
# =========================
def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/dpo_training_trl.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

def setup_wandb(config: Dict, logger: logging.Logger):
    """Setup Weights & Biases logging."""
    if not config.get("use_wandb", False):
        logger.info("Wandb logging disabled")
        return None
    
    try:
        wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            entity=config.get("wandb_entity"),
            config={
                "model": config["model_name"],
                "dataset": config["dataset_name"],
                "batch_size": config["per_device_train_batch_size"],
                "gradient_accumulation_steps": config["gradient_accumulation_steps"],
                "learning_rate": config["learning_rate"],
                "num_train_epochs": config["num_train_epochs"],
                "max_seq_length": config["max_seq_length"],
                "beta": config["beta"],
                "use_4bit": config["use_4bit"],
                "lora_r": config["lora_r"],
                "lora_alpha": config["lora_alpha"],
            },
            tags=["dpo", "qwen3", "trl", "unsloth", "4bit"]
        )
        logger.info(f"Wandb initialized: {wandb.run.url}")
        return wandb
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        return None

def get_config():
    """Get configuration for TRL DPO training."""
    config = {
        # Model configuration
        "model_name": "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        "dataset_name": "combined_pairs",
        
        # Training configuration
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-5,
        "num_train_epochs": 1,
        "max_seq_length": 1024,
        "beta": 0.1,  # DPO beta parameter
        
        # LoRA configuration
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.01,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        
        # Quantization configuration
        "use_4bit": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        
        # Output configuration
        "output_dir": "../../../models/transformers/Qwen3-1.7B/DPOTrained",
        "save_steps": 100,
        "eval_steps": 50,
        "logging_steps": 10,
        "save_total_limit": 2,
        
        # Weights & Biases configuration
        "use_wandb": True,
        "wandb_project": "qwen3-dpo-training",
        "wandb_run_name": "qwen3-1.7b-dpo-trl-unsloth",
        "wandb_entity": None,
        
        # Dataset paths
        "train_dataset_path": "dataset/combined_pairs/train",
        "eval_dataset_path": "dataset/combined_pairs/eval",
    }
    
    return config

# Load configuration
logger = setup_logging()
CONFIG = get_config()

# Create output directory if it doesn't exist
os.makedirs(CONFIG["output_dir"], exist_ok=True)
logger.info(f"Output directory: {CONFIG['output_dir']}")

# =========================
# Data Processing
# =========================
def load_and_prepare_dataset(config: Dict, logger: logging.Logger):
    """Load dataset for DPO training - directly use existing arrow datasets."""
    logger.info("Loading datasets...")
    
    # Load datasets directly (same as mini_dpo)
    train_dataset = load_from_disk(config["train_dataset_path"])
    eval_dataset = load_from_disk(config["eval_dataset_path"])
    
    logger.info(f"Loaded train dataset: {len(train_dataset)} samples")
    logger.info(f"Loaded eval dataset: {len(eval_dataset)} samples")
    
    # Validate dataset format
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        logger.info(f"Sample data keys: {list(sample.keys())}")
        if not all(key in sample for key in ["prompt", "chosen", "rejected"]):
            logger.error("Dataset missing required fields: prompt, chosen, rejected")
            sys.exit(1)
    
    return train_dataset, eval_dataset

# =========================
# Model Loading
# =========================
def load_model_and_tokenizer(config: Dict, logger: logging.Logger):
    """Load model and tokenizer using Unsloth."""
    logger.info("Loading model and tokenizer...")
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["use_4bit"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, config["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
    )
    
    # Load model and tokenizer using Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=getattr(torch, config["bnb_4bit_compute_dtype"]),
        load_in_4bit=config["use_4bit"],
        quantization_config=bnb_config if config["use_4bit"] else None,
    )
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=config["lora_target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info("Model and tokenizer loaded successfully!")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable")
    
    return model, tokenizer

# =========================
# Training Configuration
# =========================
def get_training_arguments(config: Dict):
    """Get training arguments for DPO training using Unsloth format."""
    return {
        "output_dir": config["output_dir"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "per_device_eval_batch_size": config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "num_train_epochs": config["num_train_epochs"],
        "logging_steps": config["logging_steps"],
        "save_steps": config["save_steps"],
        "eval_steps": config["eval_steps"],
        "save_total_limit": config["save_total_limit"],
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "bf16": is_bfloat16_supported(),
        "dataloader_pin_memory": False,
        "remove_unused_columns": False,
        "report_to": "wandb" if config["use_wandb"] else None,
        "run_name": config["wandb_run_name"] if config["use_wandb"] else None,
    }

def get_dpo_config(config: Dict) -> UnslothDPOConfig:
    """Get DPO configuration using Unsloth's DPOConfig."""
    return UnslothDPOConfig(
        beta=config["beta"],
        max_length=config["max_seq_length"],
        max_prompt_length=config["max_seq_length"] // 2,
        loss_type="sigmoid",
        label_smoothing=0.0,
        reference_free=False,
        padding_value=0,  # Add padding_value for Unsloth compatibility
    )

# =========================
# Main Training Function
# =========================
def main():
    """Main training function."""
    logger.info("Starting DPO training with TRL and Unsloth...")
    
    # Setup wandb logging
    wandb_logger = setup_wandb(CONFIG, logger)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(CONFIG, logger)
    
    # Load datasets
    train_dataset, eval_dataset = load_and_prepare_dataset(CONFIG, logger)
    
    # Get training configurations
    training_args = get_training_arguments(CONFIG)
    dpo_config = get_dpo_config(CONFIG)
    
    # Create DPOTrainer using Unsloth's DPOConfig
    logger.info("Creating DPOTrainer...")
    
    # Merge training arguments into DPO config
    for key, value in training_args.items():
        if hasattr(dpo_config, key):
            setattr(dpo_config, key, value)
    
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Use the same model as reference
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Log training configuration
    logger.info("Training configuration:")
    logger.info(f"  Model: {CONFIG['model_name']}")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Eval samples: {len(eval_dataset)}")
    logger.info(f"  Batch size: {CONFIG['per_device_train_batch_size']}")
    logger.info(f"  Gradient accumulation: {CONFIG['gradient_accumulation_steps']}")
    logger.info(f"  Learning rate: {CONFIG['learning_rate']}")
    logger.info(f"  Epochs: {CONFIG['num_train_epochs']}")
    logger.info(f"  Beta: {CONFIG['beta']}")
    logger.info(f"  Max length: {CONFIG['max_seq_length']}")
    
    # Start training
    logger.info("Starting training...")
    dpo_trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    dpo_trainer.save_model()
    tokenizer.save_pretrained(CONFIG["output_dir"])
    
    # Log final metrics
    if wandb_logger:
        wandb_logger.log({
            "final/model_path": CONFIG["output_dir"],
            "final/training_completed": True
        })
        wandb_logger.finish()
    
    logger.info("DPO training completed successfully!")

if __name__ == "__main__":
    main()
