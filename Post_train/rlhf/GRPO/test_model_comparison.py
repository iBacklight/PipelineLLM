"""
Test script to compare original and trained models on mathematical problems.

This script demonstrates how to use the test_model_comparison function
to evaluate the performance improvement after GRPO training.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict, Any
import json
from datetime import datetime

from grpo_algorithm import GRPOConfig, GRPOTrainer


class SimpleMathModel(nn.Module):
    """Simple model for demonstration purposes."""
    
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Simple architecture
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=512),
            num_layers=3
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        
        # Add tokenizer attribute for compatibility
        self.tokenizer = SimpleTokenizer(vocab_size)
        
    def forward(self, input_ids):
        """Forward pass through the model."""
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits
    
    def generate(self, input_ids, max_new_tokens=256, **kwargs):
        """Generate text using the model."""
        # Simple generation for demonstration
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :]
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated


class SimpleTokenizer:
    """Simple tokenizer for demonstration."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 999
        
    def __call__(self, text, return_tensors="pt", max_length=512, truncation=True, padding=True):
        """Tokenize text."""
        # Simple word-based tokenization
        words = text.split()
        tokens = [hash(word) % self.vocab_size for word in words]
        
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        if padding and len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))
        
        return {"input_ids": torch.tensor([tokens])}
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs to text."""
        # Simple decoding for demonstration
        return f"Generated response with {len(token_ids)} tokens"


def create_test_math_problems() -> List[str]:
    """Create a set of mathematical test problems."""
    return [
        "Question: What is 15 + 27?\nAnswer:",
        "Question: Solve for x: 2x + 5 = 13\nAnswer:",
        "Question: What is the area of a rectangle with length 8 and width 6?\nAnswer:",
        "Question: Calculate 3^4\nAnswer:",
        "Question: What is 144 divided by 12?\nAnswer:",
        "Question: Find the perimeter of a square with side length 9\nAnswer:",
        "Question: What is 7 × 8 + 3?\nAnswer:",
        "Question: Solve: 2x - 7 = 11\nAnswer:",
        "Question: What is the square root of 64?\nAnswer:",
        "Question: Calculate 5! (5 factorial)\nAnswer:"
    ]


def demonstrate_model_comparison():
    """Demonstrate the model comparison functionality."""
    
    print("GRPO Model Comparison Test")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create models
    print("Creating models...")
    original_model = SimpleMathModel()
    trained_model = SimpleMathModel()
    
    # Simulate some training by modifying the trained model slightly
    with torch.no_grad():
        for param in trained_model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    
    # Create GRPO configuration
    config = GRPOConfig(
        learning_rate=3e-4,
        clip_ratio=0.2,
        kl_coef=0.1,
        num_responses_per_prompt=3,
        device="cpu"
    )
    
    # Create GRPO trainer
    trainer = GRPOTrainer(trained_model, original_model, config)
    
    # Create test problems
    test_problems = create_test_math_problems()
    
    print(f"Testing on {len(test_problems)} mathematical problems")
    print()
    
    # Run comparison test
    results = trainer.test_model_comparison(
        test_prompts=test_problems,
        num_responses_per_prompt=3
    )
    
    # Display detailed results
    print("\nDetailed Results:")
    print("=" * 50)
    
    for i, (original, trained) in enumerate(zip(results['original_model'], results['trained_model'])):
        print(f"\nProblem {i+1}: {original['prompt'][:60]}...")
        print(f"  Original model:")
        print(f"    Responses: {len(original['responses'])}")
        print(f"    Avg reward: {original['avg_reward']:.3f}")
        print(f"    Max reward: {original['max_reward']:.3f}")
        print(f"  Trained model:")
        print(f"    Responses: {len(trained['responses'])}")
        print(f"    Avg reward: {trained['avg_reward']:.3f}")
        print(f"    Max reward: {trained['max_reward']:.3f}")
        print(f"  Improvement: {trained['avg_reward'] - original['avg_reward']:.3f}")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"model_comparison_results_{timestamp}.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    # Recursively convert numpy types
    def clean_results(results):
        if isinstance(results, dict):
            return {k: clean_results(v) for k, v in results.items()}
        elif isinstance(results, list):
            return [clean_results(item) for item in results]
        else:
            return convert_numpy(results)
    
    cleaned_results = clean_results(results)
    
    with open(results_file, 'w') as f:
        json.dump(cleaned_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Display summary statistics
    stats = results['comparison_stats']
    print(f"\nSummary Statistics:")
    print(f"  Original model average reward: {stats['original_mean_reward']:.3f}")
    print(f"  Trained model average reward: {stats['trained_mean_reward']:.3f}")
    print(f"  Overall improvement: {stats['improvement']:.3f}")
    print(f"  Improvement percentage: {stats['improvement_percentage']:.1f}%")
    print(f"  Problems improved: {stats['problems_improved']}/{stats['total_problems']}")
    
    return results


def test_with_custom_problems():
    """Test with custom mathematical problems."""
    
    print("\n" + "=" * 60)
    print("Custom Mathematical Problems Test")
    print("=" * 60)
    
    # Custom problems
    custom_problems = [
        "Question: What is the derivative of x^2?\nAnswer:",
        "Question: Solve the quadratic equation x^2 - 5x + 6 = 0\nAnswer:",
        "Question: What is the integral of 2x?\nAnswer:",
        "Question: Find the limit as x approaches 0 of sin(x)/x\nAnswer:",
        "Question: What is the probability of rolling a 6 on a fair die?\nAnswer:"
    ]
    
    # Create models
    original_model = SimpleMathModel()
    trained_model = SimpleMathModel()
    
    # Simulate training
    with torch.no_grad():
        for param in trained_model.parameters():
            param.add_(torch.randn_like(param) * 0.05)
    
    # Create trainer
    config = GRPOConfig(device="cpu")
    trainer = GRPOTrainer(trained_model, original_model, config)
    
    # Test
    results = trainer.test_model_comparison(
        test_prompts=custom_problems,
        num_responses_per_prompt=2
    )
    
    return results


if __name__ == "__main__":
    # Run the main demonstration
    results = demonstrate_model_comparison()
    
    # Run custom problems test
    custom_results = test_with_custom_problems()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
