import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math
try:
    from ..utils import (
        compute_advantages_and_returns,
        compute_kl_divergence,
        compute_entropy,
        compute_ppo_loss,
        compute_value_loss,
        normalize_advantages
    )
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import (
        compute_advantages_and_returns,
        compute_kl_divergence,
        compute_entropy,
        compute_ppo_loss,
        compute_value_loss,
        normalize_advantages
    )


class PolicyNetwork(nn.Module):
    """Policy network for PPO that outputs action probabilities."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection for logits
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        init_range = 0.1
        self.token_embedding.weight.data.uniform_(-init_range, init_range)
        self.position_embedding.weight.data.uniform_(-init_range, init_range)
        self.lm_head.weight.data.uniform_(-init_range, init_range)
        self.lm_head.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of the policy network.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            past_key_values: Optional cached key-value pairs for generation
        
        Returns:
            logits: Logits of shape (batch_size, seq_len, vocab_size)
            log_probs: Log probabilities of shape (batch_size, seq_len)
            new_past_key_values: Updated key-value cache
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs (clamp to max_length)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_ids = torch.clamp(position_ids, 0, self.max_length - 1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # Create attention mask for transformer (True means attend, False means ignore)
        # We need to invert the attention_mask since True means "attend" in transformer
        transformer_mask = ~attention_mask.bool()
        
        # Forward through transformer
        hidden_states = self.transformer(
            hidden_states,
            src_key_padding_mask=transformer_mask
        )
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Compute log probabilities ,for later calculate entropy and KL divergence
        log_probs = F.log_softmax(logits, dim=-1)
        
        # For simplicity, we don't implement key-value caching here
        # In a full implementation, you would cache key-value pairs for generation
        new_past_key_values = None

        #output dim: [B, T, V]
        return logits, log_probs, new_past_key_values
    
    def get_action_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Get log probabilities for specific actions.
        
        Args:
            input_ids: Input token IDs dim: [B, T_prompt]
            attention_mask: Attention mask,
            action_ids: Action token IDs to get log probs for dim: [B, action_len]
        
        Returns:
            Log probabilities for the actions
        """
        # Concatenate input and action for full sequence
        full_input_ids = torch.cat([input_ids, action_ids], dim=1)
        full_attention_mask = torch.cat([
            attention_mask,
            torch.ones_like(action_ids)
        ], dim=1)
        
        _, log_probs, _ = self.forward(full_input_ids, full_attention_mask)
        
        # Get log probabilities for the action tokens only
        # from [B, T, V] to [B, T-action_ids.shape[1], V]
        action_log_probs = log_probs[:, -action_ids.shape[1]:, :] # dim: [B, action_len, V]
        
        # Gather log probabilities for the specific action tokens
        batch_size, seq_len = action_ids.shape
        action_log_probs = torch.gather(
            action_log_probs, 
            dim=-1, # apply to last dimension
            index=action_ids.unsqueeze(-1) # index [batch_size, action_len, 1]
        ).squeeze(-1) # dim: [B, action_len], same as action_ids

        # let‘s have an example,
        # Assume：
        # B = 2 (batch)
        # action_len = 3 (total generated 3 tokens)
        # vocab_size = 5  (token candidates)
        
        # action_ids = [
        #     [1, 3, 0],  # 样本0：要选择token 1, 3, 0
        #     [2, 4, 1]   # 样本1：要选择token 2, 4, 1
        # ]
        # shape: [2, 3]

        # action_ids.unsqueeze(-1) = [
        #   [[1], [3], [0]],  # 样本0：要选择token 1, 3, 0
        #   [[2], [4], [1]]   # 样本1：要选择token 2, 4, 1
        # ]
        # shape: [2, 3, 1]

        # action_log_probs = [
        #     # 样本0的3个位置，每个位置有5个token的概率
        #     [[0.1, 0.2, 0.3, 0.4, 0.5],  # 位置0: [token0, token1, token2, token3, token4]
        #     [0.6, 0.7, 0.8, 0.9, 1.0],  # 位置1: [token0, token1, token2, token3, token4]  
        #     [1.1, 1.2, 1.3, 1.4, 1.5]], # 位置2: [token0, token1, token2, token3, token4]
            
        #     # 样本1的3个位置，每个位置有5个token的概率
        #     [[2.1, 2.2, 2.3, 2.4, 2.5],  # 位置0: [token0, token1, token2, token3, token4]
        #     [2.6, 2.7, 2.8, 2.9, 3.0],  # 位置1: [token0, token1, token2, token3, token4]
        #     [3.1, 3.2, 3.3, 3.4, 3.5]]  # 位置2: [token0, token1, token2, token3, token4]
        # ]

        # torch.gather(
        #     action_log_probs,           # [2, 3, 5] - 源张量
        #     dim=-1,                     # 在最后一个维度(词汇表维度)上操作
        #     index=action_ids.unsqueeze(-1)  # [2, 3, 1] - 索引张量
        # )

        # Process:
        #     样本0：
        #       位置0：从 [0.1, 0.2, 0.3, 0.4, 0.5] 中选择索引1 → 0.2
        #       位置1：从 [0.6, 0.7, 0.8, 0.9, 1.0] 中选择索引3 → 0.9
        #       位置2：从 [1.1, 1.2, 1.3, 1.4, 1.5] 中选择索引0 → 1.1
        #     样本1：
        #       位置0：从 [2.1, 2.2, 2.3, 2.4, 2.5] 中选择索引2 → 2.3
        #       位置1：从 [2.6, 2.7, 2.8, 2.9, 3.0] 中选择索引4 → 3.0
        #       位置2：从 [3.1, 3.2, 3.3, 3.4, 3.5] 中选择索引1 → 3.2


        # result = [
        #     [0.2, 0.9, 1.1],  # sample0：选择到的概率
        #     [2.3, 3.0, 3.2]   # sample1：选择到的概率
        # ]
        # # shape: [2, 3, 1] -> squeeze(-1) -> [2, 3]


        
        return action_log_probs


class ValueNetwork(nn.Module):
    """Value network for PPO that estimates state values."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        init_range = 0.1
        self.token_embedding.weight.data.uniform_(-init_range, init_range)
        self.position_embedding.weight.data.uniform_(-init_range, init_range)
        
        # Initialize value head
        for module in self.value_head:
            if isinstance(module, nn.Linear):
                module.weight.data.uniform_(-init_range, init_range)
                module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the value network.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
        
        Returns:
            values: State values of shape (batch_size, seq_len)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs (clamp to max_length)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_ids = torch.clamp(position_ids, 0, self.max_length - 1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # Create attention mask for transformer
        transformer_mask = ~attention_mask.bool()
        
        # Forward through transformer
        hidden_states = self.transformer(
            hidden_states,
            src_key_padding_mask=transformer_mask
        )
        
        # Compute values
        values = self.value_head(hidden_states).squeeze(-1)
        
        return values


class PPOPolicy:
    """PPO Policy that combines policy and value networks."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_length: int = 512,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        self.device = device
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Initialize networks
        self.policy_net = PolicyNetwork(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=max_length,
            dropout=dropout
        ).to(device)
        
        self.value_net = ValueNetwork(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=max_length,
            dropout=dropout
        ).to(device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=3e-4,
            eps=1e-5
        )
        
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=3e-4,
            eps=1e-5
        )
    
    def get_action_and_value(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action log probabilities and values for given states.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            action_log_probs: Log probabilities of actions
            values: State values
            entropy: Policy entropy
        """
        # Get policy outputs
        _, log_probs, _ = self.policy_net(input_ids, attention_mask)
        
        # Get value estimates
        values = self.value_net(input_ids, attention_mask)
        
        # Compute entropy
        entropy = compute_entropy(log_probs)
        
        return log_probs, values, entropy
    
    def evaluate_actions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            action_ids: Action token IDs to evaluate
        
        Returns:
            action_log_probs: Log probabilities of the actions
            values: State values
            entropy: Policy entropy
        """
        # Get action log probabilities
        action_log_probs = self.policy_net.get_action_log_probs(
            input_ids, attention_mask, action_ids
        )
        
        # Get value estimates
        values = self.value_net(input_ids, attention_mask)
        
        # Compute entropy
        _, log_probs, _ = self.policy_net(input_ids, attention_mask)
        entropy = compute_entropy(log_probs)
        
        return action_log_probs, values, entropy
    
    def update(
        self,
        experiences: List[Dict[str, torch.Tensor]],
        clip_range: float = 0.2,
        value_clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4
    ) -> Dict[str, float]:
        """
        Update the policy and value networks using PPO.
        
        Args:
            experiences: List of experience dictionaries
            clip_range: PPO clipping range
            value_clip_range: Value function clipping range
            entropy_coef: Entropy coefficient
            value_coef: Value function loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            num_epochs: Number of training epochs
        
        Returns:
            Dictionary of training metrics
        """
        # Convert experiences to tensors
        input_ids = torch.cat([exp['input_ids'] for exp in experiences])
        attention_mask = torch.cat([exp['attention_mask'] for exp in experiences])
        action_ids = torch.cat([exp['action_ids'] for exp in experiences])
        old_log_probs = torch.cat([exp['log_probs'] for exp in experiences])
        old_values = torch.cat([exp['values'] for exp in experiences])
        advantages = torch.cat([exp['advantages'] for exp in experiences])
        returns = torch.cat([exp['returns'] for exp in experiences])
        
        # Normalize advantages
        advantages = normalize_advantages(advantages)
        
        # Training metrics
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'kl_divergence': 0.0,
            'clip_ratio': 0.0
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Get current policy outputs
            action_log_probs, values, entropy = self.evaluate_actions(
                input_ids, attention_mask, action_ids
            )
            
            # Take mean over action sequence length for policy loss
            action_log_probs_mean = action_log_probs.mean(dim=1)  # (batch_size,)  apply mean to each sample at action_len dimension
            old_log_probs_mean = old_log_probs.mean(dim=1)  # (batch_size,)  apply mean to each sample at action_len dimension
            
            # Compute policy loss
            policy_loss = compute_ppo_loss(
                old_log_probs_mean, action_log_probs_mean, advantages, clip_range
            )
            
            # Compute value loss (use last value for each sequence)
            values_last = values[:, -1]  # (batch_size,)  apply last value to each sample at action_len dimension
            old_values_last = old_values[:, -1]  # (batch_size,)  apply last value to each sample at action_len dimension
            value_loss = compute_value_loss(
                old_values_last, values_last, returns, value_clip_range
            )
            
            # Compute entropy loss
            entropy_loss = -entropy_coef * entropy.mean() 
            
            # Total loss
            total_loss = policy_loss + value_coef * value_loss + entropy_loss
            
            # Update policy (actor)
            self.policy_optimizer.zero_grad()
            total_loss.backward(retain_graph=True) # maintain comput graph to kepp grads of policy and value loss
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_grad_norm)
            self.policy_optimizer.step()
            
            # Update value function (critic)
            value_policy_loss = policy_loss + value_coef * value_loss # only need policy and value loss
            self.value_optimizer.zero_grad()
            value_policy_loss.backward() # grab the grad maintained by total_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_grad_norm)
            self.value_optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                kl_div = compute_kl_divergence(old_log_probs_mean, action_log_probs_mean)
                ratio = torch.exp(action_log_probs_mean - old_log_probs_mean)
                clip_ratio = ((ratio - 1.0).abs() > clip_range).float().mean()
                
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy_loss'] += entropy_loss.item()
                metrics['kl_divergence'] += kl_div.item()
                metrics['clip_ratio'] += clip_ratio.item()
        
        # Average metrics over epochs
        for key in metrics:
            metrics[key] /= num_epochs
        
        return metrics
    
    def save(self, filepath: str):
        """Save the policy and value networks."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load the policy and value networks."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
