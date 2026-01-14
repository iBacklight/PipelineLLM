# TRL Examples: PPO and GRPO

This directory contains simple examples of using the TRL (Transformer Reinforcement Learning) library for RLHF training.

## Overview

| Algorithm | File | Description |
|-----------|------|-------------|
| PPO | `ppo_trl_example.py` | Proximal Policy Optimization with value head |
| GRPO | `grpo_trl_example.py` | Group Relative Policy Optimization |

## Installation

```bash
pip install trl transformers torch peft accelerate datasets
```

## PPO (Proximal Policy Optimization)

PPO is the classic RLHF algorithm used in training models like ChatGPT.

### Key Features:
- Uses a **value head** to estimate expected rewards
- Maintains a **reference model** for KL penalty
- **Clipping mechanism** to constrain policy updates

### Architecture:
```
┌─────────────────────────────────────────────────────────────┐
│                        PPO Training                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Policy Model          Reference Model        Value Head    │
│   (trainable)           (frozen)               (trainable)   │
│        │                     │                      │        │
│        ▼                     ▼                      ▼        │
│   Generate Response    Compute KL Penalty    Estimate Value  │
│        │                     │                      │        │
│        └─────────────────────┴──────────────────────┘        │
│                              │                               │
│                              ▼                               │
│                      Compute Advantage                       │
│                      A = R - V(s)                            │
│                              │                               │
│                              ▼                               │
│                      PPO Clipped Loss                        │
│            L = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Run:
```bash
python ppo_trl_example.py
```

## GRPO (Group Relative Policy Optimization)

GRPO is a more recent algorithm that simplifies PPO by removing the need for a value model.

### Key Features:
- **No value model needed** - uses group-relative advantages
- Generates **multiple responses per prompt** (group)
- Computes advantages **relative to group mean**
- Works well with **verifiable rewards** (e.g., math, code)

### Architecture:
```
┌─────────────────────────────────────────────────────────────┐
│                       GRPO Training                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   For each prompt, generate N responses (group):             │
│                                                              │
│   Prompt: "What is 2+2?"                                     │
│      │                                                       │
│      ├──► Response 1: "The answer is 4"     → Reward: 1.0   │
│      ├──► Response 2: "2+2 equals 4"        → Reward: 1.0   │
│      ├──► Response 3: "It's 5"              → Reward: 0.0   │
│      └──► Response 4: "The result is 4"     → Reward: 1.0   │
│                                                              │
│   Group mean reward: μ = 0.75                                │
│   Group std: σ = 0.43                                        │
│                                                              │
│   Normalized advantages:                                     │
│      A₁ = (1.0 - 0.75) / 0.43 = +0.58                       │
│      A₂ = (1.0 - 0.75) / 0.43 = +0.58                       │
│      A₃ = (0.0 - 0.75) / 0.43 = -1.74                       │
│      A₄ = (1.0 - 0.75) / 0.43 = +0.58                       │
│                                                              │
│   Policy gradient update using these advantages              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### GRPO Loss Function:
```
L_GRPO = -E[Σᵢ (Âᵢ · log π(yᵢ|x)) - β · KL(π || π_ref)]

where:
- Âᵢ = (rᵢ - μ_group) / σ_group  (group-normalized advantage)
- β = KL penalty coefficient
- π_ref = reference policy (frozen)
```

### Run:
```bash
python grpo_trl_example.py
```

## Comparison: PPO vs GRPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| Value Model | Required | Not required |
| Memory Usage | Higher (value head + ref model) | Lower |
| Sample Efficiency | Moderate | Higher (group normalization) |
| Best For | General RLHF | Verifiable rewards (math, code) |
| Complexity | Higher | Lower |
| Variance | Higher | Lower (group normalization) |

## Key TRL Classes

```python
# PPO
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

# GRPO
from trl import GRPOConfig, GRPOTrainer

# Common
from trl import RewardTrainer, DPOTrainer, SFTTrainer
```

## Tips for Training

1. **Start with a small model** for debugging
2. **Use LoRA** for parameter-efficient training
3. **Monitor KL divergence** - too high means the model is drifting too far
4. **Reward scaling** matters - normalize rewards if they vary a lot
5. **Group size** in GRPO affects variance - larger groups = more stable

## References

- [TRL Documentation](https://huggingface.co/docs/trl)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [GRPO Paper (DeepSeek-R1)](https://arxiv.org/abs/2501.12948)

