#!/usr/bin/env python3
"""
Main script to run Reward Model training with different configurations.

run like this:
python run_training.py --config qwen3_0.6b --8bit_optim --bf16 --gradient_checkpointing --use_wandb --wandb_project "my-reward-model"

# Maximum memory optimization (all features):
python run_training.py --config qwen3_0.6b --8bit_optim --8bit_model --bf16 --gradient_checkpointing --use_wandb --wandb_project "my-reward-model"

Tested on Qwen3-0.6B with NVIDIA GeForce RTX 4080 GPU
"""

import os
import sys
import argparse
import logging
from typing import Optional
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config, create_custom_config, CONFIGS
from train_rm import RewardModelTrainer
from evaluate_reward_model import RewardModelEvaluation
from reward_model_utils import load_preference_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Reward Model Training")
    
    # Configuration selection
    parser.add_argument(
        "--config", 
        type=str, 
        default="qwen3_0.6b",
        choices=list(CONFIGS.keys()) + ["custom"],
        help="Configuration to use"
    )
    
    # Custom configuration parameters
    parser.add_argument("--model_name", type=str, help="Model name for custom config")
    parser.add_argument("--dataset_name", type=str, help="Dataset name for custom config")
    parser.add_argument("--batch_size", type=int, help="Batch size for custom config")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for custom config")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs for custom config")
    parser.add_argument("--max_train_samples", type=int, help="Max training samples for custom config")
    parser.add_argument("--output_dir", type=str, help="Output directory for custom config")
    
    # Training options
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="reward-model-training", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, help="W&B entity name")
    
    # Memory optimization
    parser.add_argument("--8bit_optim", action="store_true", help="Use 8-bit AdamW optimizer")
    parser.add_argument("--8bit_model", action="store_true", help="Use 8-bit model quantization")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Loss configuration
    parser.add_argument("--loss_type", type=str, default="ranking", 
                       choices=["ranking", "mse", "bce"], 
                       help="Loss type for reward model training")
    
    # Evaluation options
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--model_path", type=str, help="Path to trained model for evaluation")
    
    args = parser.parse_args()
    
    # Get configuration
    if args.config == "custom":
        # Create custom configuration
        custom_params = {}
        if args.model_name:
            custom_params["model_name"] = args.model_name
        if args.dataset_name:
            custom_params["dataset_name"] = args.dataset_name
        if args.batch_size:
            custom_params["batch_size"] = args.batch_size
        if args.learning_rate:
            custom_params["learning_rate"] = args.learning_rate
        if args.num_epochs:
            custom_params["num_epochs"] = args.num_epochs
        if args.max_train_samples:
            custom_params["max_train_samples"] = args.max_train_samples
        if args.output_dir:
            custom_params["output_dir"] = args.output_dir
        
        config = create_custom_config(**custom_params)
    else:
        config = get_config(args.config)
    
    # Override with command line arguments
    if args.use_wandb:
        config.use_wandb = True
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_entity:
        config.wandb_entity = args.wandb_entity
    if args.seed:
        config.seed = args.seed
    
    # Memory optimization overrides
    if args.__dict__.get('8bit_optim', False):
        config.use_8bit_optimizer = True
    if args.__dict__.get('8bit_model', False):
        config.use_8bit_model = True
    if args.gradient_checkpointing:
        config.gradient_checkpointing = True
    if args.__dict__.get('bf16', False):
        config.use_bf16 = True
    
    # Loss configuration override
    if args.loss_type:
        config.loss_type = args.loss_type
    
    # Print configuration
    logger.info("Training Configuration:")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Dataset: {config.dataset_name}")
    logger.info(f"Batch Size: {config.batch_size}")
    logger.info(f"Learning Rate: {config.learning_rate}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Max Train Samples: {config.max_train_samples}")
    logger.info(f"Output Dir: {config.output_dir}")
    logger.info(f"Use W&B: {config.use_wandb}")
    logger.info(f"8-bit Optimizer: {config.use_8bit_optimizer}")
    logger.info(f"8-bit Model: {config.use_8bit_model}")
    logger.info(f"Gradient Checkpointing: {config.gradient_checkpointing}")
    logger.info(f"bf16 Precision: {config.use_bf16}")
    logger.info(f"Loss Type: {config.loss_type}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    if args.eval_only:
        # Run evaluation only
        if not args.model_path:
            logger.error("Model path required for evaluation")
            return
        
        
        evaluator = RewardModelEvaluation(
            args.model_path,
            config.model_name,
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load test data
        
        test_data = load_preference_dataset(config.dataset_name, "test")
        
        # Generate evaluation report
        report = evaluator.generate_evaluation_report(
            test_data, 
            os.path.join(config.output_dir, "evaluation_report.json")
        )
        
        logger.info("Evaluation completed!")
        
    else:
        # Run training
        trainer = RewardModelTrainer(config)
        
        # Load data
        train_data, eval_data = trainer.load_data()
        
        # Train model
        trainer.train(train_data, eval_data)
        
        logger.info("Training completed!")
        
        # Run evaluation
        logger.info("Running evaluation...")
        
        
        model_path = os.path.join(config.output_dir, "final_model")
        evaluator = RewardModelEvaluation(
            model_path,
            config.model_name,
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Generate evaluation report
        report = evaluator.generate_evaluation_report(
            eval_data, 
            os.path.join(config.output_dir, "evaluation_report.json")
        )
        
        logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
