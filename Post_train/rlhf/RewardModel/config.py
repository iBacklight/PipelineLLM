#!/usr/bin/env python3
"""
Configuration file for Reward Model training.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class RewardModelConfig:
    """Configuration for Reward Model training."""
    
    # Model configuration
    model_name: str = "Qwen/Qwen3-0.6B"  # Base model
    max_length: int = 512
    freeze_layers: int = 0  # Number of layers to freeze
    
    # Training configuration
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data configuration
    dataset_name: str = "yitingxie/rlhf-reward-datasets"
    train_split: str = "train"
    eval_split: str = "test"
    max_train_samples: int = 10000
    max_eval_samples: int = 1000
    
    # Training schedule
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    
    # Output configuration
    output_dir: str = "./reward_model_checkpoints"
    run_name: Optional[str] = None
    
    # Logging and monitoring
    use_wandb: bool = False
    wandb_project: str = "reward-model-training"
    wandb_entity: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    
    # Memory optimization
    use_8bit_optimizer: bool = False  # 8-bit AdamW optimizer
    use_8bit_model: bool = False      # 8-bit model quantization
    gradient_checkpointing: bool = False
    use_bf16: bool = False            # Use bf16 precision for training
    
    # Loss configuration
    loss_type: str = "ranking"        # Loss type: "ranking", "mse", "bce"
    
    # Evaluation
    eval_datasets: List[str] = None
    
    def __post_init__(self):
        if self.eval_datasets is None:
            self.eval_datasets = [
                "yitingxie/rlhf-reward-datasets",
                "Anthropic/hh-rlhf",
                "Dahoas/rm-static",
                "Dahoas/full-hh-rlhf"
            ]
        
        if self.run_name is None:
            self.run_name = f"reward_model_{self.model_name.split('/')[-1]}"


# Predefined configurations for different scenarios
CONFIGS = {
    "qwen3_0.6b": RewardModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        batch_size=8,
        learning_rate=5e-6,
        num_epochs=3,
        max_train_samples=5000,
        max_eval_samples=500,
        freeze_layers=0
    ),
    
    "qwen3_1.7b": RewardModelConfig(
        model_name="Qwen/Qwen3-1.7B",
        batch_size=4,
        learning_rate=3e-6,
        num_epochs=3,
        max_train_samples=5000,
        max_eval_samples=500,
        freeze_layers=0
    )
}


def get_config(config_name: str = "qwen3_0.6b") -> RewardModelConfig:
    """Get a predefined configuration."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(CONFIGS.keys())}")
    
    return CONFIGS[config_name]


def create_custom_config(**kwargs) -> RewardModelConfig:
    """Create a custom configuration."""
    return RewardModelConfig(**kwargs)
