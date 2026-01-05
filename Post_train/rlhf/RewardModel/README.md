# Reward Model Training with Qwen3 Models

This directory contains a complete implementation for training Reward Models using Hugging Face's official methods and datasets, based on Qwen3-0.6B and Qwen3-1.7B models.

## Features

- **Hugging Face Integration**: Uses official Hugging Face datasets and training methods
- **Qwen3 Base Model**: Built on Qwen3-0.6B and Qwen3-1.7B models for efficient training
- **Preference Learning**: Trains on human preference data (chosen vs rejected responses)
- **Comprehensive Evaluation**: Multiple evaluation metrics and analysis
- **Configurable**: Easy-to-use configuration system with predefined setups
- **Monitoring**: Optional Weights & Biases integration

## Files

- `train_reward_model_complete.py`: Main training script
- `reward_model_utils.py`: Utility functions and model classes
- `evaluate_reward_model.py`: Evaluation script
- `config.py`: Configuration management
- `run_training.py`: Main entry point with predefined configurations
- `README.md`: This documentation

## Quick Start

### 1. Basic Training

```bash
# Train with default configuration (Qwen3 0.6B, small dataset)
python run_training.py --config qwen3_0.6b_small

# Train with medium configuration
python run_training.py --config qwen3_0.6b_medium

# Train with custom parameters
python run_training.py --config custom \
    --model_name "Qwen/Qwen3-0.6B" \
    --batch_size 16 \
    --learning_rate 3e-6 \
    --num_epochs 5
```

### 2. Training with Weights & Biases

```bash
python run_training.py --config qwen3_0.6b_medium --use_wandb --wandb_project "my-reward-model"
```

### 3. Evaluation Only

```bash
python run_training.py --eval_only --model_path "./reward_model_checkpoints/final_model"
```

## Available Configurations

| Configuration | Model | Batch Size | Learning Rate | Epochs | Max Train Samples | Description |
|---------------|-------|------------|---------------|--------|-------------------|-------------|
| `qwen3_0.6b` | Qwen3 0.6B | 8 | 5e-6 | 3 | 5,000 | Standard training |
| `qwen3_1.7b` | Qwen3 1.7B | 4 | 3e-6 | 3 | 5,000 | Larger model training |

### Memory Optimization Arguments
- `--8bit_optim`: Use 8-bit AdamW optimizer
- `--8bit_model`: Use 8-bit model quantization  
- `--gradient_checkpointing`: Enable gradient checkpointing
- `--bf16`: Use bf16 precision for training

## Supported Datasets

- **Anthropic/hh-rlhf**: Human preference data from Anthropic
- **Dahoas/rm-static**: Static reward model dataset
- **Dahoas/full-hh-rlhf**: Full HH-RLHF dataset

## Model Architecture

The Reward Model is built on top of Qwen models with:

- **Base Model**: Qwen3 0.6B or 1.7B transformer
- **Reward Head**: 2-layer MLP (hidden_size → hidden_size//2 → 1)
- **Pooling**: Mean pooling over sequence length
- **Freezing**: Optional layer freezing for efficiency

## Memory Optimization

### 8-bit Optimizer
- **8-bit AdamW**: Reduces optimizer memory usage by ~50%
- **Training Efficiency**: Enables larger batch sizes
- **Compatibility**: Works with any model (quantized or not)

### 8-bit Model Quantization
- **Model Weights**: 8-bit quantized model weights
- **Memory Efficiency**: Reduces model memory footprint
- **Automatic Device Mapping**: Efficient GPU memory allocation

### Gradient Checkpointing
- **Memory vs Compute Trade-off**: Saves memory by recomputing gradients
- **Configurable**: Enable/disable based on available memory
- **Automatic Detection**: Works with supported model architectures

### bf16 Precision
- **Memory Efficiency**: Reduces memory usage by ~50% compared to fp32
- **Better Stability**: More stable than fp16 for training
- **Hardware Support**: Requires modern GPUs (RTX 30/40 series, A100, etc.)
- **Automatic Detection**: Falls back to fp32 if not supported

### Usage Examples
```bash
# Standard training
python run_training.py --config qwen3_0.6b

# 8-bit optimizer only (model stays in full precision)
python run_training.py --config qwen3_0.6b --8bit_optim --gradient_checkpointing

# 8-bit model quantization only (optimizer stays standard)
python run_training.py --config qwen3_0.6b --8bit_model --gradient_checkpointing

# Both 8-bit optimizer and model quantization
python run_training.py --config qwen3_0.6b --8bit_optim --8bit_model --gradient_checkpointing

# bf16 precision training
python run_training.py --config qwen3_0.6b --bf16 --gradient_checkpointing

# Maximum memory optimization (all features)
python run_training.py --config qwen3_0.6b --8bit_optim --8bit_model --bf16 --gradient_checkpointing

# Large model with memory optimization
python run_training.py --config qwen3_1.7b --8bit_optim --8bit_model --bf16 --gradient_checkpointing
```

### Memory Requirements
| Configuration | GPU Memory | Batch Size | Speed | Notes |
|---------------|------------|------------|-------|-------|
| Qwen3-0.6B (standard) | ~8-10GB | 8 | 100% | Full precision |
| Qwen3-0.6B (bf16) | ~4-5GB | 16 | 95-100% | bf16 precision |
| Qwen3-0.6B (8-bit optimizer) | ~6-8GB | 12 | 90-95% | Optimizer only |
| Qwen3-0.6B (8-bit model) | ~4-6GB | 16 | 85-90% | Model only |
| Qwen3-0.6B (8-bit full) | ~3-5GB | 16 | 80-85% | Both |
| Qwen3-0.6B (all optimizations) | ~2-3GB | 16 | 75-80% | bf16 + 8-bit + checkpointing |
| Qwen3-1.7B (standard) | ~16-20GB | 4 | 100% | Full precision |
| Qwen3-1.7B (bf16) | ~8-10GB | 8 | 95-100% | bf16 precision |
| Qwen3-1.7B (8-bit optimizer) | ~12-16GB | 6 | 90-95% | Optimizer only |
| Qwen3-1.7B (8-bit model) | ~8-10GB | 8 | 85-90% | Model only |
| Qwen3-1.7B (8-bit full) | ~6-8GB | 8 | 80-85% | Both |
| Qwen3-1.7B (all optimizations) | ~4-5GB | 8 | 75-80% | bf16 + 8-bit + checkpointing |

## Training Process

1. **Data Loading**: Load preference data from Hugging Face datasets
2. **Preprocessing**: Tokenize chosen and rejected responses
3. **Training**: Train reward model to predict higher rewards for chosen responses
4. **Evaluation**: Evaluate on preference accuracy and reward distribution
5. **Saving**: Save model, tokenizer, and evaluation results

## Evaluation Metrics

- **Preference Accuracy**: Percentage of correctly predicted preferences
- **Reward Distribution**: Mean, std, min, max of predicted rewards
- **Reward Difference**: Difference between chosen and rejected rewards
- **Pattern Analysis**: Correlation between rewards and response length

## Usage Examples

### Custom Configuration

```python
from config import create_custom_config

config = create_custom_config(
    model_name="Qwen/Qwen3-0.6B",
    batch_size=16,
    learning_rate=3e-6,
    num_epochs=5,
    max_train_samples=10000,
    use_wandb=True
)
```

### Direct Training

```python
from train_reward_model_complete import RewardModelTrainer
from config import get_config

config = get_config("qwen3_0.6b_medium")
trainer = RewardModelTrainer(config)

train_data, eval_data = trainer.load_data()
trainer.train(train_data, eval_data)
```

### Evaluation

```python
from evaluate_reward_model import RewardModelEvaluation

evaluator = RewardModelEvaluation(
    model_path="./reward_model_checkpoints/final_model",
    model_name="Qwen/Qwen3-0.6B"
)

# Load test data
from reward_model_utils import load_preference_dataset
test_data = load_preference_dataset("Anthropic/hh-rlhf", "test")

# Generate evaluation report
report = evaluator.generate_evaluation_report(test_data)
```

## Output Structure

```
reward_model_checkpoints/
├── final_model/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── best_model_step_XXX/
├── checkpoint_step_XXX/
└── evaluation_report.json
```

## Installation

```bash
(python -m) pip install torch transformers datasets numpy tqdm scikit-learn matplotlib seaborn
(python -m) pip install wandb  # Optional
```

## Memory Requirements

| Configuration | GPU Memory | RAM |
|---------------|------------|-----|
| qwen3_0.6b_small | ~4GB | ~8GB |
| qwen3_0.6b_medium | ~6GB | ~12GB |
| qwen3_0.6b_large | ~10GB | ~20GB |
| qwen3_1.7b_small | ~6GB | ~12GB |
| qwen3_1.7b_medium | ~10GB | ~20GB |
| qwen3_1.7b_large | ~12GB | ~24GB |

## Example Output

### Single Input
```python
# Input: "What is the capital of France? The capital is Paris."
output = {
    "rewards": tensor([2.3456])
}
```

### Batch Input
```python
# Input: ["Question 1...", "Question 2...", "Question 3..."]
output = {
    "rewards": tensor([2.3456, -0.1234, 1.7890])
}
```

### Training Output
```python
# During training with labels
output = {
    "rewards": tensor([2.3456, -0.1234, 1.7890]),
    "loss": tensor(0.1234)
}
```

## Tips for Training

1. **Start Small**: Use `qwen3_0.6b_small` for initial testing
2. **Monitor Metrics**: Use W&B to track training progress
3. **Adjust Learning Rate**: Lower for larger models or datasets
4. **Freeze Layers**: Freeze early layers for faster training
5. **Batch Size**: Increase batch size if you have more GPU memory

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Freeze more layers
- Use smaller model

### Slow Training
- Increase batch size
- Freeze more layers
- Use mixed precision training
- Reduce max_length

### If Meet Poor Performance
- Increase training data
- Adjust learning rate
- Train for more epochs
- Check data quality


## License

This implementation follows the same license as the base Qwen models.
