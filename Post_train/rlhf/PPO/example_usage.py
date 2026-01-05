#!/usr/bin/env python3
"""
Example usage of the PPO implementation for LLM RLHF.

This script demonstrates how to use the PPO trainer with a simple dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import argparse
from typing import List, Dict, Any

from ..utils import set_seed, RewardModel
from .ppo_algorithm import PPOPolicy
from .ppo_trainer import PPOTrainer


class SimpleRLHFDataset(Dataset):
    """Simple dataset for RLHF training."""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input
        input_text = item['input']
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'input_text': input_text
        }


class SimpleTokenizer:
    """Simple tokenizer for demonstration purposes."""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.unk_token_id = 1
        
        # Simple word-to-id mapping
        self.word_to_id = {
            '<pad>': 0,
            '<unk>': 1,
            '<eos>': 2,
            '<sos>': 3
        }
        
        # Add some common words
        common_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'shall', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
            'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'
        ]
        
        for i, word in enumerate(common_words):
            self.word_to_id[word] = i + 4
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        words = text.lower().split()
        token_ids = [self.word_to_id.get(word, self.unk_token_id) for word in words]
        return token_ids
    
    def __call__(self, text, max_length=None, padding=None, truncation=None, return_tensors=None):
        """Tokenize text."""
        token_ids = self.encode(text)
        
        if max_length:
            if truncation:
                token_ids = token_ids[:max_length]
            if padding == 'max_length':
                while len(token_ids) < max_length:
                    token_ids.append(self.pad_token_id)
        
        result = {
            'input_ids': torch.tensor([token_ids]),
            'attention_mask': torch.tensor([[1 if tid != self.pad_token_id else 0 for tid in token_ids]])
        }
        
        if return_tensors == 'pt':
            return result
        return result


def create_sample_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Create sample data for training."""
    sample_inputs = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot.",
        "How do you make chocolate cake?",
        "What are the benefits of exercise?",
        "Describe the process of photosynthesis.",
        "What is artificial intelligence?",
        "How does the internet work?",
        "What are renewable energy sources?",
        "Explain the water cycle."
    ]
    
    data = []
    for i in range(num_samples):
        data.append({
            'input': sample_inputs[i % len(sample_inputs)],
            'id': i
        })
    
    return data


def create_simple_reward_model(vocab_size: int, hidden_size: int = 256) -> RewardModel:
    """Create a simple reward model for demonstration."""
    return RewardModel(hidden_size=hidden_size, vocab_size=vocab_size)


def main():
    parser = argparse.ArgumentParser(description='PPO RLHF Training Example')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./ppo_example_checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of training samples')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=10000)
    
    # Create sample data
    train_data = create_sample_data(args.num_samples)
    eval_data = create_sample_data(20)  # Smaller eval set
    
    # Create datasets
    train_dataset = SimpleRLHFDataset(train_data, tokenizer, args.max_length)
    eval_dataset = SimpleRLHFDataset(eval_data, tokenizer, args.max_length)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create policy
    policy = PPOPolicy(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        max_length=args.max_length,
        device=args.device
    )
    
    # Create reward model
    reward_model = create_simple_reward_model(tokenizer.vocab_size, args.hidden_size)
    reward_model.to(args.device)
    
    # Create trainer
    trainer = PPOTrainer(
        policy=policy,
        reward_model=reward_model,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        log_freq=5,
        save_freq=20
    )
    
    print(f"Starting PPO training with {args.num_episodes} episodes...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")
    
    # Train
    trainer.train(
        train_dataloader=train_dataloader,
        num_episodes=args.num_episodes,
        save_checkpoints=True,
        log_to_file=True
    )
    
    # Evaluate
    print("\nEvaluating final policy...")
    eval_metrics = trainer.evaluate(eval_dataloader, num_eval_episodes=5)
    
    print("\nFinal Evaluation Metrics:")
    for key, value in eval_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, "final_model.pt")
    policy.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
