"""
PPO implementation for LLM RLHF.

This package contains a minimal implementation of Proximal Policy Optimization
for Large Language Model Reinforcement Learning from Human Feedback using only PyTorch.
"""

from .ppo_algorithm import PPOPolicy, PolicyNetwork, ValueNetwork
from .ppo_trainer import PPOTrainer

__all__ = [
    'PPOPolicy',
    'PolicyNetwork', 
    'ValueNetwork',
    'PPOTrainer'
]
