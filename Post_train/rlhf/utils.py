import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class Experience:
    """Single experience for RLHF training."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_ids: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ExperienceBuffer:
    """Buffer to store and manage RLHF experiences."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: List[Experience] = []
    
    def add(self, experience: Experience):
        """Add a single experience to the buffer."""
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def add_batch(self, experiences: List[Experience]):
        """Add multiple experiences to the buffer."""
        for exp in experiences:
            self.add(exp)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


def compute_advantages_and_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    use_gae: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages and returns using GAE (Generalized Advantage Estimation).
    
    Args:
        rewards: Tensor of shape (batch_size, seq_len) containing rewards
        values: Tensor of shape (batch_size, seq_len) containing value estimates
        gamma: Discount factor
        lam: GAE lambda parameter
        use_gae: Whether to use GAE or simple returns
    
    Returns:
        advantages: Computed advantages of shape (batch_size, seq_len)
        returns: Computed returns of shape (batch_size, seq_len)
    """
    batch_size, seq_len = rewards.shape
    
    if not use_gae:
        # Simple returns: R_t = r_t
        returns = rewards.clone()
        advantages = returns - values
        return advantages, returns
    
    # GAE computation
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    # Compute TD errors: δ_t = r_t + γV(s_{t+1}) - V(s_t)
    # For the last timestep, there's no next state, so V(s_{t+1}) = 0
    next_values = torch.cat([values[:, 1:], torch.zeros(batch_size, 1, device=values.device)], dim=1)
    td_errors = rewards + gamma * next_values - values
    
    # Compute advantages using GAE (vectorized)
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            # Last timestep: A_t = δ_t
            advantages[:, t] = td_errors[:, t]
        else:
            # GAE: A_t = δ_t + γλA_{t+1}
            advantages[:, t] = td_errors[:, t] + gamma * lam * advantages[:, t + 1]
    
    # Compute returns: R_t = A_t + V(s_t)
    returns = advantages + values
    
    return advantages, returns


def compute_kl_divergence(
    log_probs_old: torch.Tensor,
    log_probs_new: torch.Tensor
) -> torch.Tensor:
    """Compute KL divergence between old and new policy distributions."""
    return (log_probs_old - log_probs_new).mean()


def compute_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Compute entropy of the policy distribution."""
    # log_prob dim: [B, T, V]
    # exp(log_prob) dim: [B, T, V]
    # (log_probs * torch.exp(log_probs)) dim: [B, T, V]
    # .sum(dim=-1) dim: [B, T]
    # .mean() dim: [B]
    return -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()


def clip_advantages(advantages: torch.Tensor, clip_range: float = 0.2) -> torch.Tensor:
    """Clip advantages to prevent extreme values."""
    return torch.clamp(advantages, -clip_range, clip_range)


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages to have zero mean and unit variance."""
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def compute_reward_to_go(
    rewards: torch.Tensor,
    gamma: float = 0.99
) -> torch.Tensor:
    """Compute reward-to-go for each timestep."""
    returns = torch.zeros_like(rewards)
    running_return = 0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns


def pad_sequences(
    sequences: List[torch.Tensor],
    pad_token_id: int = 0,
    max_length: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of token sequences
        pad_token_id: Token ID to use for padding
        max_length: Maximum length (if None, use max length in batch)
    
    Returns:
        padded_sequences: Padded tensor of shape (batch_size, max_length)
        attention_mask: Attention mask tensor of shape (batch_size, max_length)
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    batch_size = len(sequences)
    padded_sequences = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_length)
        padded_sequences[i, :seq_len] = seq[:seq_len]
        attention_mask[i, :seq_len] = 1
    
    return padded_sequences, attention_mask


def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """Create attention mask from input IDs."""
    return (input_ids != pad_token_id).long()


def compute_ppo_loss(
    log_probs_old: torch.Tensor,
    log_probs_new: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2
) -> torch.Tensor:
    """
    Compute PPO policy loss with clipping.
    
    Args:
        log_probs_old: Log probabilities from old policy
        log_probs_new: Log probabilities from new policy
        advantages: Advantage estimates
        clip_range: Clipping range for PPO
    
    Returns:
        PPO loss
    """
    ratio = torch.exp(log_probs_new - log_probs_old)
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    
    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean() # dim： scalar
    
    return policy_loss


def compute_value_loss(
    values_old: torch.Tensor,
    values_new: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float = 0.2
) -> torch.Tensor:
    """
    Compute PPO value loss with clipping.
    
    Args:
        values_old: Value estimates from old policy
        values_new: Value estimates from new policy
        returns: Target returns
        clip_range: Clipping range for PPO
    
    Returns:
        Value loss
    """
    value_pred_clipped = values_old + torch.clamp(
        values_new - values_old, -clip_range, clip_range
    )
    
    value_loss = torch.max(
        F.mse_loss(values_new, returns, reduction='none'),
        F.mse_loss(value_pred_clipped, returns, reduction='none')
    ).mean()
    
    return value_loss # dim: scalar


class RewardModel(nn.Module):
    """Simple reward model for RLHF."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True),
            num_layers=6
        )
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute rewards."""
        x = self.embedding(input_ids)
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        
        # Pool over sequence length
        pooled = x.mean(dim=1)
        rewards = self.reward_head(pooled)
        return rewards.squeeze(-1)


def load_reward_data(file_path: str) -> List[Dict[str, Any]]:
    """Load reward data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_reward_data(data: List[Dict[str, Any]], file_path: str):
    """Save reward data to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def compute_metrics(
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor
) -> Dict[str, float]:
    """Compute training metrics."""
    return {
        'mean_reward': rewards.mean().item(),
        'std_reward': rewards.std().item(),
        'mean_advantage': advantages.mean().item(),
        'std_advantage': advantages.std().item(),
        'mean_return': returns.mean().item(),
        'std_return': returns.std().item(),
    }


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
