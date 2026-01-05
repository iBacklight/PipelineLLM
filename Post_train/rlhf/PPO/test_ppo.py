#!/usr/bin/env python3
"""
Simple test script to verify the PPO implementation works correctly.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import set_seed, RewardModel
from PPO.ppo_algorithm import PPOPolicy
from PPO.ppo_trainer import PPOTrainer


def test_policy_creation():
    """Test that policy can be created successfully."""
    print("Testing policy creation...")
    
    policy = PPOPolicy(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        max_length=32,
        device="cpu"
    )
    
    print("✓ Policy created successfully")
    return policy


def test_policy_forward():
    """Test policy forward pass."""
    print("Testing policy forward pass...")
    
    policy = PPOPolicy(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        max_length=32,
        device="cpu"
    )
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    log_probs, values, entropy = policy.get_action_and_value(input_ids, attention_mask)
    
    assert log_probs.shape == (batch_size, seq_len, 1000), f"Expected log_probs shape {(batch_size, seq_len, 1000)}, got {log_probs.shape}"
    assert values.shape == (batch_size, seq_len), f"Expected values shape {(batch_size, seq_len)}, got {values.shape}"
    assert entropy.dim() == 0, f"Expected scalar entropy, got shape {entropy.shape}"
    
    print("✓ Policy forward pass successful")


def test_reward_model():
    """Test reward model."""
    print("Testing reward model...")
    
    reward_model = RewardModel(hidden_size=64, vocab_size=1000)
    
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    rewards = reward_model(input_ids, attention_mask)
    
    assert rewards.shape == (batch_size,), f"Expected rewards shape {(batch_size,)}, got {rewards.shape}"
    
    print("✓ Reward model test successful")


def test_experience_collection():
    """Test experience collection."""
    print("Testing experience collection...")
    
    policy = PPOPolicy(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        max_length=32,
        device="cpu"
    )
    
    reward_model = RewardModel(hidden_size=64, vocab_size=1000)
    
    trainer = PPOTrainer(
        policy=policy,
        reward_model=reward_model,
        device="cpu",
        max_length=32,
        batch_size=2
    )
    
    # Create dummy input
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones(2, 10)
    
    # Collect experiences
    experiences = trainer.collect_experiences(input_ids, attention_mask, num_responses=1)
    
    assert len(experiences) == 1, f"Expected 1 experience, got {len(experiences)}"
    
    exp = experiences[0]
    assert hasattr(exp, 'input_ids'), "Experience missing input_ids"
    assert hasattr(exp, 'response_ids'), "Experience missing response_ids"
    assert hasattr(exp, 'rewards'), "Experience missing rewards"
    assert hasattr(exp, 'advantages'), "Experience missing advantages"
    
    print("✓ Experience collection successful")


def test_training_step():
    """Test a single training step."""
    print("Testing training step...")
    
    policy = PPOPolicy(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        max_length=32,
        device="cpu"
    )
    
    reward_model = RewardModel(hidden_size=64, vocab_size=1000)
    
    trainer = PPOTrainer(
        policy=policy,
        reward_model=reward_model,
        device="cpu",
        max_length=32,
        batch_size=2
    )
    
    # Create dummy input
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones(2, 10)
    
    # Collect experiences
    experiences = trainer.collect_experiences(input_ids, attention_mask, num_responses=1)
    
    # Train step
    metrics = trainer.train_step(experiences)
    
    assert isinstance(metrics, dict), "Training step should return metrics dict"
    assert 'policy_loss' in metrics, "Metrics missing policy_loss"
    assert 'value_loss' in metrics, "Metrics missing value_loss"
    
    print("✓ Training step successful")


def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    
    # Create dummy dataset
    input_ids = torch.randint(0, 1000, (10, 20))
    attention_mask = torch.ones(10, 20)
    
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Test iteration
    for batch in dataloader:
        batch_input_ids, batch_attention_mask = batch
        assert batch_input_ids.shape[0] <= 4, "Batch size too large"
        break
    
    print("✓ Data loading successful")


def main():
    """Run all tests."""
    print("Running PPO implementation tests...\n")
    
    # Set seed for reproducibility
    set_seed(42)
    
    try:
        test_policy_creation()
        test_policy_forward()
        test_reward_model()
        test_experience_collection()
        test_training_step()
        test_data_loading()
        
        print("\n🎉 All tests passed! PPO implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
