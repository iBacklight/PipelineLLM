"""
GRPO (Group Relative Policy Optimization) Package

This package provides a comprehensive implementation of the GRPO algorithm,
a variant of PPO that uses group-based relative policy optimization.

Main components:
- GRPOConfig: Configuration class for GRPO parameters
- GRPOTrainer: Core GRPO training algorithm
- GRPOTrainingManager: High-level training interface
- GRPOBuffer: Experience buffer for training data
- GroupManager: Handles experience grouping strategies

Example usage:
    from grpo_algorithm import GRPOConfig, GRPOTrainer
    from grpo_trainer import GRPOTrainingManager, create_grpo_config
    
    # Create configuration
    config = create_grpo_config("CartPole-v1")
    
    # Create training manager
    trainer = GRPOTrainingManager(config, "CartPole-v1")
    
    # Train the agent
    results = trainer.train(total_episodes=1000)
"""

from .grpo_algorithm import (
    GRPOConfig,
    GRPOTrainer,
    GRPOBuffer,
    GroupManager,
    create_simple_policy_net,
    create_simple_value_net
)

from .grpo_trainer import (
    GRPOTrainingManager,
    GRPOEnvironmentWrapper,
    create_grpo_config
)

__version__ = "1.0.0"
__author__ = "GRPO Implementation"
__email__ = "grpo@example.com"

__all__ = [
    "GRPOConfig",
    "GRPOTrainer", 
    "GRPOBuffer",
    "GroupManager",
    "GRPOTrainingManager",
    "GRPOEnvironmentWrapper",
    "create_simple_policy_net",
    "create_simple_value_net",
    "create_grpo_config"
]

