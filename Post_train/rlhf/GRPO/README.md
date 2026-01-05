# Group Relative Policy Optimization (GRPO) for LLM Training

This repository contains a corrected implementation of the Group Relative Policy Optimization (GRPO) algorithm specifically designed for Large Language Model (LLM) training scenarios.

## Key Features

### ✅ Corrected Implementation
- **Grouping by Prompt**: Responses are grouped based on the same input prompt, not using k-means clustering
- **Direct Advantage Computation**: Advantages are computed directly from rewards without Generalized Advantage Estimation (GAE)
- **Simplified Architecture**: No group manager or complex grouping strategies needed
- **LLM-Focused**: Designed specifically for language model training scenarios

### ✅ Loss Function Components
- **Clipped Surrogate Objective**: Prevents large policy updates for training stability
- **KL Divergence Penalty**: Keeps the policy close to the reference model
- **No Value Function**: Eliminates the need for a separate critic network

## Algorithm Overview

GRPO is a variant of PPO that groups multiple responses generated from the same prompt and computes advantages relative to the group performance. This approach:

1. **Generates multiple responses** for each input prompt
2. **Groups responses** by their originating prompt
3. **Computes advantages** by normalizing rewards within each group
4. **Updates policy** using token-level importance sampling with clipped surrogate objective and KL divergence penalty

### Key Implementation Details

- **Token-Level Computation**: Importance sampling ratios and KL divergence are computed for each token, not averaged at the sample level
- **Group-Level Advantages**: Advantages are computed by normalizing rewards within each prompt group
- **Mixed-Level Loss**: Combines token-level ratios with group-level advantages for proper GRPO training
- **GSM8K Ground Truth Integration**: Extracts ground truth from GSM8K answer column using `#### <number>` format
- **Hybrid Reward System**: Combines correctness-based rewards (when answer extraction succeeds) with quality-based rewards (fallback)

## Mathematical Formulation

The GRPO loss function is:

```
L(θ) = E[min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)] - β · D_KL(π_θ || π_ref)
```

Where:
- `r_t(θ)` is the probability ratio between new and old policies
- `A_t` is the advantage computed from group-normalized rewards
- `ε` is the clipping parameter
- `β` is the KL divergence coefficient
- `D_KL(π_θ || π_ref)` is the KL divergence using the DeepSeek formula:

```
D_KL(π_θ || π_ref) = (π_ref(o_i|q) / π_θ(o_i|q)) - log(π_ref(o_i|q) / π_θ(o_i|q)) - 1
```

## File Structure

```
GRPO/
├── grpo_algorithm.py          # Core GRPO algorithm implementation
├── grpo_trainer.py            # LLM training framework
├── grpo_qwen3_training.py     # Qwen3-0.6B + GSM8K training script
├── example_grpo.py            # Basic demonstration script
├── example_qwen3_grpo.py      # Qwen3 + GSM8K example
├── test_model_comparison.py   # Full model comparison test
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Usage

### Installation

```bash
# For GPU support, ensure you have CUDA installed
# For 4-bit quantization, install bitsandbytes
pip install bitsandbytes
```

### Basic Usage

```python
from grpo_algorithm import GRPOConfig, GRPOTrainer
from grpo_trainer import LLMGRPOTrainer, create_grpo_config_for_llm

# Create configuration
config = create_grpo_config_for_llm(
    learning_rate=3e-4,
    clip_ratio=0.2,
    kl_coef=0.1,
    num_responses_per_prompt=4
)

# Create trainer
trainer = LLMGRPOTrainer(model, ref_model, tokenizer, config)

# Train
results = trainer.train(training_prompts, num_epochs=100)
```

### Qwen3-0.6B + GSM8K Training

```python
from grpo_qwen3_training import Qwen3GRPOTrainer, GRPOQwenConfig

# Create configuration for Qwen3 training
config = GRPOQwenConfig(
    model_name="Qwen/Qwen2.5-0.5B",
    max_train_samples=1000,
    batch_size=8,
    num_responses_per_prompt=4,
    lora_r=16,
    lora_alpha=32,
    load_in_4bit=True
)

# Create trainer
trainer = Qwen3GRPOTrainer(config)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./grpo_qwen3_model")
```

### Quick Start

```bash
# Run basic example
python example_grpo.py

# Run Qwen3 + GSM8K example with wandb logging
python grpo_training.py

# Test model comparison (simulation)
python test_comparison_logic.py
```

### Model Comparison Testing

The GRPO implementation includes a comprehensive model comparison test function:

```python
from grpo_algorithm import GRPOTrainer

# After training your model
trainer = GRPOTrainer(trained_model, original_model, config)

# Test on mathematical problems
test_problems = [
    "Question: What is 15 + 27?\nAnswer:",
    "Question: Solve for x: 2x + 5 = 13\nAnswer:",
    "Question: What is the area of a rectangle with length 8 and width 6?\nAnswer:"
]

# Run comparison test
results = trainer.test_model_comparison(
    test_prompts=test_problems,
    num_responses_per_prompt=3
)

# Analyze results
print(f"Improvement: {results['comparison_stats']['improvement']:.3f}")
print(f"Improvement %: {results['comparison_stats']['improvement_percentage']:.1f}%")
```

The test function provides:
- **Side-by-side comparison** of original vs trained model
- **Mathematical reasoning evaluation** with detailed scoring
- **Statistical analysis** including improvement metrics
- **Individual problem results** with before/after scores
- **JSON export** for further analysis

### Reward Function Design

The GRPO implementation uses a sophisticated hybrid reward function for GSM8K mathematical reasoning:

#### Ground Truth Extraction
- **Source**: Extracts ground truth from GSM8K answer column using `#### <number>` format
- **Robustness**: Handles multiple answer formats with fallback patterns
- **Consistency**: Uses same extraction function for both ground truth and model responses

#### Reward Components
1. **Correctness Reward** (0.6 for correct, 0.2 for wrong when extraction succeeds)
2. **Quality Fallback** (0.3 base when extraction fails)
3. **Length Bonus** (0.2 for detailed responses >50 chars)
4. **Mathematical Content** (0.1 per math indicator, max 0.5)
5. **Step Reasoning** (0.3 for structured thinking keywords)
6. **Format Compliance** (0.15 for `####` format, 0.1 for others)
7. **Numerical Work** (0.1 for number presence)
8. **Calculation Evidence** (0.1 for equations/calculations)

#### Prompt Engineering
```python
prompt = f"""Question: {question}

Please solve this step by step and provide your final answer in the format "#### [number]".

Answer:"""
```

This encourages models to use the standard GSM8K format for reliable answer extraction.

### Weights & Biases (wandb) Integration

The GRPO training script includes comprehensive wandb logging for experiment tracking:

#### Logged Metrics

**Training Metrics:**
- `train/total_loss` - Overall GRPO loss
- `train/policy_loss` - Policy gradient loss  
- `train/kl_penalty` - KL divergence penalty
- `train/kl_mean/std` - KL divergence statistics
- `train/ratio_mean/std` - Importance sampling ratio statistics
- `train/advantages_mean/std` - Advantage statistics

**Reward Metrics:**
- `rewards/mean/std/max/min/median` - Reward distribution statistics
- `rewards/q25/q75` - Reward quartiles

**Group Statistics:**
- `group_stats/num_groups` - Number of prompt groups
- `group_stats/avg_group_size` - Average responses per group

**Evaluation Metrics:**
- `eval/avg_reward` - Average evaluation reward
- `eval/avg_response_length` - Response quality metrics
- `eval/avg_math_score` - Mathematical content analysis

**Visualizations:**
- Sample responses table with prompts and rewards
- Reward histograms and distributions
- Model architecture monitoring

#### Usage

```python
# Enable wandb logging
config = GRPOQwenConfig(
    use_wandb=True,
    wandb_project="my-grpo-experiment",
    wandb_run_name="experiment-1"
)

# Train with logging
trainer = Qwen3GRPOTrainer(config)
trainer.train()  # Automatically logs to wandb
```

#### Setup

```bash
# Install wandb
pip install wandb

# Login to wandb
wandb login

# Run training with logging
python grpo_training.py
```

### Configuration Parameters

- `learning_rate`: Learning rate for the optimizer
- `clip_ratio`: Clipping parameter for the surrogate objective
- `kl_coef`: Coefficient for KL divergence penalty
- `num_responses_per_prompt`: Number of responses to generate per prompt
- `batch_size`: Batch size for training
- `advantage_normalization`: Whether to normalize advantages within groups

## Key Differences from Standard PPO

1. **No Value Function**: GRPO eliminates the need for a separate value function
2. **Group-Based Advantages**: Advantages are computed relative to group performance
3. **Prompt-Based Grouping**: Responses are grouped by their input prompt
4. **Simplified Architecture**: No complex grouping strategies or group managers

## Example

Run the demonstration script to see GRPO in action:

```bash
python example_grpo.py
```

This will show:
- How responses are grouped by prompt
- How advantages are computed from group rewards
- How the loss function combines clipping and KL divergence
- Training statistics and group performance

## Requirements

- PyTorch
- NumPy
- Python 3.7+

## References

- Group Relative Policy Optimization (GRPO) - A variant of PPO for LLM training
- Focuses on group-based relative evaluation for more stable training
- Eliminates the need for value functions in policy optimization
