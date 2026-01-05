# PPO for LLM RLHF

This directory contains a minimal implementation of Proximal Policy Optimization (PPO) for Large Language Model Reinforcement Learning from Human Feedback (RLHF) using only PyTorch.

## Features

- **Pure PyTorch Implementation**: No dependencies on TRL or Unsloth libraries
- **Modular Design**: Separate policy and value networks
- **Shared Utilities**: Common functions that can be used by other RLHF algorithms (like GRPO)
- **Configurable**: Easy to customize hyperparameters and model architecture
- **Example Usage**: Complete example script demonstrating training

## Files

- `ppo_algorithm.py`: Core PPO algorithm with policy and value networks
- `ppo_trainer.py`: Training loop and experience collection
- `example_usage.py`: Example script showing how to use the implementation
- `README.md`: This documentation file

## Dependencies

- PyTorch
- NumPy
- tqdm (for progress bars)
- json (for logging)

## Quick Start

1. **Basic Usage**:
```python
from ppo_algorithm import PPOPolicy
from ppo_trainer import PPOTrainer
from utils import RewardModel

# Create policy
policy = PPOPolicy(
    vocab_size=10000,
    hidden_size=512,
    num_layers=6,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Create reward model
reward_model = RewardModel(hidden_size=512, vocab_size=10000)

# Create trainer
trainer = PPOTrainer(
    policy=policy,
    reward_model=reward_model,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Train
trainer.train(train_dataloader, num_episodes=1000)
```

2. **Run Example**:
```bash
python example_usage.py --num_episodes 100 --batch_size 8 --device cuda
```

## Architecture

### Policy Network
- Transformer-based architecture
- Outputs action probabilities for token generation
- Supports both sampling and greedy decoding

### Value Network
- Separate transformer for value estimation
- Estimates state values for advantage computation
- Uses GAE (Generalized Advantage Estimation)

### PPO Algorithm
- Clipped policy loss
- Clipped value loss
- Entropy regularization
- KL penalty (optional)

## Key Components

### Experience Buffer
Stores and manages training experiences:
- Input sequences
- Generated responses
- Log probabilities
- Value estimates
- Rewards and advantages

### Reward Computation
- Supports custom reward models
- KL penalty for reference policy divergence
- Configurable reward shaping

### Training Loop
- Experience collection through response generation
- PPO updates with configurable hyperparameters
- Logging and checkpointing
- Evaluation metrics

## Hyperparameters

### PPO Parameters
- `clip_range`: PPO clipping range (default: 0.2)
- `value_clip_range`: Value function clipping (default: 0.2)
- `entropy_coef`: Entropy regularization coefficient (default: 0.01)
- `value_coef`: Value function loss coefficient (default: 0.5)
- `gamma`: Discount factor (default: 0.99)
- `lam`: GAE lambda parameter (default: 0.95)

### Training Parameters
- `batch_size`: Training batch size
- `num_epochs`: Number of PPO update epochs
- `max_grad_norm`: Gradient clipping norm
- `learning_rate`: Learning rate for optimizers

## Customization

### Custom Reward Model
```python
class MyRewardModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        # Your reward model architecture
        
    def forward(self, input_ids, attention_mask):
        # Your reward computation
        return rewards

# Use in trainer
trainer = PPOTrainer(policy=policy, reward_model=MyRewardModel(...))
```

### Custom Tokenizer
The example uses a simple tokenizer, but you can replace it with any tokenizer:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Use with your dataset
```

### Custom Dataset
```python
class MyRLHFDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        # Your dataset implementation
        
    def __getitem__(self, idx):
        # Return input_ids, attention_mask, etc.
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
```

## Monitoring Training

The trainer provides comprehensive logging:
- Episode rewards
- Policy and value losses
- KL divergence
- Clipping ratios
- Response lengths

Checkpoints are saved regularly and can be loaded:
```python
# Load checkpoint
policy.load("checkpoint_episode_100.pt")
```

## Integration with Other Algorithms

The shared utilities in `utils.py` are designed to be reusable:
- Experience buffer
- Advantage computation
- Loss functions
- Metrics computation

These can be used by other RLHF algorithms like GRPO, DPO, etc.

## Limitations

This is a minimal implementation for educational purposes:
- No advanced generation strategies (beam search, nucleus sampling, etc.)
- Simple reward model
- Basic tokenizer
- No distributed training support

For production use, consider:
- Using pre-trained language models
- Implementing more sophisticated reward models
- Adding distributed training support
- Optimizing for memory efficiency

## Example Output

```
Episode 0: Reward: 0.1500, Policy Loss: 0.2341, Value Loss: 0.1234, KL Div: 0.0123, Clip Ratio: 0.0500
Episode 5: Reward: 0.1800, Policy Loss: 0.1987, Value Loss: 0.0987, KL Div: 0.0156, Clip Ratio: 0.0450
...
```

## Contributing

Feel free to extend this implementation:
- Add more sophisticated generation strategies
- Implement additional RLHF algorithms
- Improve the reward model
- Add more evaluation metrics
