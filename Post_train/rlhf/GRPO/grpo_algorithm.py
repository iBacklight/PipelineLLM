"""
Group Relative Policy Optimization (GRPO) Algorithm Implementation

GRPO is a variant of PPO specifically designed for LLM training that:
1. Groups responses by the same prompt (no k-means clustering)
2. Computes advantages directly from rewards (no GAE)
3. Uses clipped surrogate objective with KL divergence penalty
4. Eliminates the need for a separate value function

Key features:
- Multiple responses per prompt are grouped together
- Advantages are computed by normalizing rewards within each group
- Loss function includes clipping and KL divergence terms
- No group manager or complex grouping strategies needed

IMPORTANT COMPUTATION LEVELS IN GRPO:
=====================================

1. ADVANTAGES: GROUP LEVEL
   - Each group contains multiple responses to the same prompt
   - Advantages are computed by normalizing rewards within each group
   - This ensures fair comparison between responses to the same prompt
   - Formula: A_i = (R_i - μ_group) / σ_group

2. RATIO: TOKEN LEVEL  
   - Probability ratios are computed for each token in each response
   - This is the standard PPO ratio computation at token level
   - Formula: ratio_i = exp(log π_θ(token_i) - log π_old(token_i))
   - Each token gets its own importance sampling ratio

3. KL DIVERGENCE: TOKEN LEVEL
   - KL divergence is computed for each token using DeepSeek formula
   - Each token gets its own KL divergence value
   - Formula: D_KL = (π_ref/π_θ) - log(π_ref/π_θ) - 1

4. LOSS: MIXED LEVEL
   - Combines token-level ratios with group-level advantages
   - Policy loss: -min(ratio * advantage, clipped_ratio * advantage)
   - KL penalty: β * mean(KL_divergence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
import logging
from collections import defaultdict
import math


@dataclass
class GRPOConfig:
    """Configuration class for GRPO algorithm."""
    
    # Learning parameters
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    kl_coef: float = 0.1  # KL divergence coefficient
    max_grad_norm: float = 0.5
    
    # Training parameters
    num_epochs: int = 4
    batch_size: int = 64
    gamma: float = 0.99  # Not used in advantage computation, kept for compatibility
    
    # GRPO specific parameters
    num_responses_per_prompt: int = 4  # Number of responses to generate per prompt
    advantage_normalization: bool = True  # Whether to normalize advantages within groups
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.kl_coef < 0:
            raise ValueError("KL coefficient must be non-negative")
        if self.clip_ratio <= 0 or self.clip_ratio >= 1:
            raise ValueError("Clip ratio must be between 0 and 1")
        if self.num_responses_per_prompt < 2:
            raise ValueError("Number of responses per prompt must be at least 2")


class GRPOTrainer:
    """GRPO Trainer implementation for LLM training."""
    
    def __init__(self, 
                 policy_net: nn.Module,
                 ref_policy_net: nn.Module,
                 config: GRPOConfig):
        """
        Initialize GRPO trainer.
        
        Args:
            policy_net: Policy network (the model being trained)
            ref_policy_net: Reference policy network (frozen, for KL divergence)
            config: GRPO configuration
        """
        self.policy_net = policy_net
        self.ref_policy_net = ref_policy_net
        self.config = config
        
        # Freeze reference policy
        for param in self.ref_policy_net.parameters():
            param.requires_grad = False
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=config.learning_rate
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.training_stats = defaultdict(list)
        
    def compute_advantages(self, 
                          rewards: torch.Tensor,
                          group_indices: List[int]) -> torch.Tensor:
        """
        Compute advantages for each response based on group rewards.
        
        IMPORTANT: This function computes advantages at GROUP LEVEL.
        - Each group contains multiple responses to the same prompt
        - Advantages are computed by normalizing rewards within each group
        - This is different from token-level computation
        
        Args:
            rewards: Reward tensor for each response (response-level rewards)
            group_indices: List indicating which group each response belongs to
            
        Returns:
            advantages: Computed advantages for each response (group-normalized)
        """
        advantages = torch.zeros_like(rewards)
        
        # GROUP LEVEL COMPUTATION: Group rewards by prompt
        # Each group contains responses generated from the same prompt
        unique_groups = list(set(group_indices))
        
        for group_id in unique_groups:
            # Find all responses belonging to this group (same prompt)
            group_mask = torch.tensor([gid == group_id for gid in group_indices], 
                                    device=rewards.device)
            group_rewards = rewards[group_mask]
            
            if len(group_rewards) == 0:
                continue
            
            # GROUP LEVEL STATISTICS: Compute group mean and std
            # This is the key difference - we normalize within groups, not globally
            group_mean = group_rewards.mean()  # Mean reward for this prompt group
            group_std = group_rewards.std()    # Std reward for this prompt group
            
            if self.config.advantage_normalization and group_std > 0:
                # GROUP LEVEL NORMALIZATION: Normalize advantages within the group
                # This ensures fair comparison between responses to the same prompt
                group_advantages = (group_rewards - group_mean) / (group_std + 1e-8)
            else:
                # Use raw rewards as advantages (still group-relative)
                group_advantages = group_rewards - group_mean
            
            # Assign group-normalized advantages back to individual responses
            advantages[group_mask] = group_advantages
        
        return advantages
    
    def compute_kl_divergence(self, 
                             log_probs: torch.Tensor,
                             ref_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between current and reference policy.
        Using the DeepSeek paper formula:
        D_KL(π_θ || π_ref) = (π_ref(o_i|q) / π_θ(o_i|q)) - log(π_ref(o_i|q) / π_θ(o_i|q)) - 1
        
        IMPORTANT: This function computes KL divergence at TOKEN LEVEL.
        - Each token in each response gets its own KL divergence value
        - This is different from response-level or group-level computation
        
        Args:
            log_probs: Log probabilities from current policy (token-level)
            ref_log_probs: Log probabilities from reference policy (token-level)
            
        Returns:
            kl_div: KL divergence for each token (token-level)
        """
        # TOKEN LEVEL CONVERSION: Convert log probabilities to probabilities
        # Each element corresponds to a token in the response
        probs = torch.exp(log_probs)        # π_θ(o_i|q) for each token i
        ref_probs = torch.exp(ref_log_probs) # π_ref(o_i|q) for each token i
        
        # TOKEN LEVEL RATIO: Compute ratio for each token
        # ratio[i] = π_ref(o_i|q) / π_θ(o_i|q) for token i
        ratio = ref_probs / (probs + 1e-8)  # Add small epsilon to avoid division by zero
        
        # TOKEN LEVEL KL DIVERGENCE: DeepSeek formula applied to each token
        # Each token gets its own KL divergence value
        kl_div = ratio - torch.log(ratio + 1e-8) - 1
        
        return kl_div
    
    def compute_loss(self, 
                    log_probs: List[torch.Tensor],
                    ref_log_probs: List[torch.Tensor],
                    advantages: torch.Tensor,
                    old_log_probs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute GRPO loss with clipping and KL divergence.
        
        IMPORTANT COMPUTATION LEVELS:
        - RATIO: Computed at TOKEN LEVEL (for each token in each response)
        - ADVANTAGES: Computed at GROUP LEVEL (normalized within prompt groups)
        - KL DIVERGENCE: Computed at TOKEN LEVEL (for each token)
        
        Args:
            log_probs: List of token-level log probabilities for each response
            ref_log_probs: List of token-level reference log probabilities for each response
            advantages: Computed advantages (response-level, group-normalized)
            old_log_probs: List of token-level old log probabilities for each response
            
        Returns:
            Dictionary containing loss components
        """
        # Flatten token-level log probabilities for computation
        # Handle both 0-d tensors and empty tensors properly
        valid_log_probs = [lp for lp in log_probs if len(lp) > 0]
        valid_ref_log_probs = [rlp for rlp in ref_log_probs if len(rlp) > 0]
        valid_old_log_probs = [olp for olp in old_log_probs if len(olp) > 0]
        # valid_log_probs = []
        # for lp in log_probs:
        #     if isinstance(lp, torch.Tensor):
        #         if lp.dim() > 0 and lp.numel() > 0:
        #             valid_log_probs.append(lp)
        #     elif len(lp) > 0:
        #         valid_log_probs.append(lp)
        
        # valid_ref_log_probs = []
        # for rlp in ref_log_probs:
        #     if isinstance(rlp, torch.Tensor):
        #         if rlp.dim() > 0 and rlp.numel() > 0:
        #             valid_ref_log_probs.append(rlp)
        #     elif len(rlp) > 0:
        #         valid_ref_log_probs.append(rlp)
        
        # valid_old_log_probs = []
        # for olp in old_log_probs:
        #     if isinstance(olp, torch.Tensor):
        #         if olp.dim() > 0 and olp.numel() > 0:
        #             valid_old_log_probs.append(olp)
        #     elif len(olp) > 0:
        #         valid_old_log_probs.append(olp)
        
        # if not valid_log_probs or not valid_ref_log_probs or not valid_old_log_probs:
        #     # Return zero loss if no valid log probabilities
        #     return {
        #         'total_loss': torch.tensor(0.0, device=self.config.device),
        #         'policy_loss': torch.tensor(0.0, device=self.config.device),
        #         'kl_penalty': torch.tensor(0.0, device=self.config.device),
        #         'ratio_mean': torch.tensor(1.0, device=self.config.device),
        #         'ratio_std': torch.tensor(0.0, device=self.config.device),
        #         'kl_mean': torch.tensor(0.0, device=self.config.device),
        #         'kl_std': torch.tensor(0.0, device=self.config.device),
        #         'advantages_mean': advantages.mean(),
        #         'advantages_std': advantages.std()
        #     }
        
        all_log_probs = torch.cat(valid_log_probs)
        all_ref_log_probs = torch.cat(valid_ref_log_probs)
        all_old_log_probs = torch.cat(valid_old_log_probs)
        
        # Create response-level advantages tensor that matches token count
        response_advantages = []
        for i, (lp, adv) in enumerate(zip(log_probs, advantages)):
            # Handle both 0-d tensors and empty tensors properly
            if isinstance(lp, torch.Tensor):
                if lp.dim() > 0 and lp.numel() > 0:
                    response_advantages.extend([adv] * lp.numel())
            elif len(lp) > 0:
                response_advantages.extend([adv] * len(lp))
        
        if not response_advantages:
            # Fallback if no valid advantages
            response_advantages = torch.tensor([0.0], device=self.config.device)
        else:
            response_advantages = torch.tensor(response_advantages, device=self.config.device)
        
        # TOKEN LEVEL COMPUTATION: Compute probability ratios
        # ratio[i] = exp(log_prob[i] - old_log_prob[i]) for each token i
        ratio = torch.exp(all_log_probs - all_old_log_probs)
        
        # TOKEN LEVEL CLIPPING: Clipped surrogate objective
        clipped_ratio = torch.clamp(ratio, 
                                  1 - self.config.clip_ratio,
                                  1 + self.config.clip_ratio)
        
        # TOKEN LEVEL POLICY LOSS: 
        # Each token gets its own ratio and advantage
        policy_loss = -torch.min(
            ratio * response_advantages,
            clipped_ratio * response_advantages
        ).mean()
        
        # TOKEN LEVEL KL DIVERGENCE: 
        # KL divergence is computed at token level using DeepSeek formula
        kl_div = self.compute_kl_divergence(all_log_probs, all_ref_log_probs)
        kl_penalty = self.config.kl_coef * kl_div.mean()
        
        # Total loss combines token-level and group-level components
        total_loss = policy_loss + kl_penalty
        
        # Additional statistics for monitoring
        ratio_mean = ratio.mean()
        ratio_std = ratio.std()
        kl_mean = kl_div.mean()
        kl_std = kl_div.std()
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'kl_penalty': kl_penalty,
            'ratio_mean': ratio_mean,
            'ratio_std': ratio_std,
            'kl_mean': kl_mean,
            'kl_std': kl_std,
            'advantages_mean': advantages.mean(),
            'advantages_std': advantages.std()
        }
    
    def update(self, 
              prompts: List[str],
              responses: List[str],
              rewards: List[float],
              group_indices: List[int]) -> Dict[str, float]:
        """
        Update policy using GRPO.
        
        Args:
            prompts: List of input prompts
            responses: List of generated responses
            rewards: List of rewards for each response
            group_indices: List indicating which group each response belongs to
            
        Returns:
            Training statistics
        """
        if len(responses) == 0:
            return {}
        
        # Convert to tensors
        rewards_tensor = torch.FloatTensor(rewards).to(self.config.device)
        
        # Compute advantages
        advantages = self.compute_advantages(rewards_tensor, group_indices)
        
        # Normalize advantages globally
        if self.config.advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass through current policy (for old_log_probs - no gradients needed)
        # Note: old_log_probs represent the policy at data collection time (frozen)
        with torch.no_grad():
            current_log_probs = self._get_log_probs(prompts, responses, use_ref=False)
        
        # Forward pass through current policy (for new policy - needs gradients)
        # Note: new_log_probs are used for ratio computation and need gradients
        new_log_probs = self._get_log_probs(prompts, responses, use_ref=False)
        
        # Forward pass through reference policy (frozen - no gradients)
        # Note: ref_log_probs are used for KL divergence and should be frozen
        ref_log_probs = self._get_log_probs(prompts, responses, use_ref=True)
        
        # Compute loss
        loss_dict = self.compute_loss(
            new_log_probs, 
            ref_log_probs, 
            advantages, 
            current_log_probs
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            self.config.max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        
        # Convert to float for logging
        stats = {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in loss_dict.items()}
        
        return stats
    
    def _get_log_probs(self, 
                      prompts: List[str], 
                      responses: List[str], 
                      use_ref: bool = False) -> torch.Tensor:
        """
        Get log probabilities for given prompts and responses using Qwen3-0.6B.
        
        This method:
        1. Tokenizes prompts and responses using Qwen3 tokenizer
        2. Computes log probabilities for response tokens only
        3. Returns average log probability per response (token-level computation)
        
        Args:
            prompts: List of input prompts
            responses: List of generated responses
            use_ref: Whether to use reference policy (frozen, no gradients)
            
        Returns:
            log_probs: List of token-level log probabilities for each response
        """
        import torch.nn.functional as F
        
        # Get the appropriate model
        policy_net = self.ref_policy_net if use_ref else self.policy_net
        
        # Set model mode appropriately
        if use_ref:
            # Reference model should always be in eval mode
            policy_net.eval()
        else:
            # Current model mode depends on calling context
            # (gradients handled by calling function)
            pass
        
        all_log_probs = []
        
        for prompt, response in zip(prompts, responses):
            try:
                # Combine prompt and response
                full_text = prompt + response
                
                # Use tokenizer from trainer (not from model)
                tokenizer = getattr(self, 'tokenizer', None)
                if tokenizer is None:
                    # Fallback: try to get tokenizer from model
                    tokenizer = getattr(policy_net, 'tokenizer', None)
                
                if tokenizer is None:
                    self.logger.warning("No tokenizer available, using fallback")
                    all_log_probs.append(torch.tensor([0.0], device=self.config.device))
                    continue
                
                inputs = tokenizer(
                    full_text,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                    padding=True
                ).to(self.config.device)
                
                response_inputs = tokenizer(
                    response,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                    padding=True
                ).to(self.config.device)
                
                input_ids = inputs.input_ids
                response_ids = response_inputs.input_ids[0]
                
                if len(input_ids[0]) == 0 or len(response_ids) == 0: # if the input or response is empty, return 0
                    all_log_probs.append(torch.tensor(0.0, device=self.config.device))
                    continue
                
                # Forward pass through the model
                with torch.cuda.amp.autocast(enabled=False):  # Disable autocast to save memory
                    outputs = policy_net(input_ids)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Get log probabilities for all tokens
                log_probs_all = F.log_softmax(logits, dim=-1) # shape: [1, T, V]
                
                # Find response tokens in the full sequence
                response_start_idx = len(input_ids[0]) - len(response_ids)
                response_log_probs = []
                
                for i, token_id in enumerate(response_ids):
                    if response_start_idx + i < len(input_ids[0]):
                        # Get log probability for this token
                        # response_start_idx + i - 1 is the index of the lasttoken in the full sequence
                        token_log_prob = log_probs_all[0, response_start_idx + i - 1, token_id] # log prob for the token based on the last position prediction
                        response_log_probs.append(token_log_prob)
                
                if response_log_probs:
                    # Return token-level log probabilities for GRPO
                    # Each token gets its own log probability for proper IS computation
                    token_log_probs = torch.stack(response_log_probs)
                    all_log_probs.append(token_log_probs)
                else:
                    # Return empty tensor if no valid tokens
                    all_log_probs.append(torch.tensor([], device=self.config.device))
                
                # Clear intermediate variables to save memory
                del outputs, logits, log_probs_all
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                # Fallback to simple computation if tokenizer fails
                self.logger.warning(f"Error in log prob computation: {e}")
                all_log_probs.append(torch.tensor(0.0, device=self.config.device))
        
        # Return list of token-level log probabilities for each response
        # Each element is a tensor of log probabilities for tokens in that response
        return all_log_probs
    
    def _tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize text using a simple tokenizer.
        In practice, this would use the actual Qwen3 tokenizer.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
        """
        # Simple word-based tokenization for demonstration
        # In practice, you would use: tokenizer.encode(text, return_tensors="pt")
        words = text.split()
        # Convert words to simple token IDs (hash-based)
        tokens = [hash(word) % 1000 for word in words]
        return tokens
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get training statistics."""
        return dict(self.training_stats)
    
    def test_model_comparison(self, 
                             test_prompts: List[str], 
                             num_responses_per_prompt: int = 3) -> Dict[str, any]:
        """
        Test and compare original and trained models on mathematical problems.
        
        Args:
            test_prompts: List of mathematical problem prompts
            num_responses_per_prompt: Number of responses to generate per prompt
            
        Returns:
            Dictionary containing comparison results
        """
        self.logger.info(f"Testing model comparison on {len(test_prompts)} mathematical problems")
        
        results = {
            'original_model': [],
            'trained_model': [],
            'comparison_stats': {}
        }
        
        for i, prompt in enumerate(test_prompts):
            self.logger.info(f"Testing problem {i+1}/{len(test_prompts)}: {prompt[:100]}...")
            
            # Test original model (reference)
            original_responses = []
            original_rewards = []
            for _ in range(num_responses_per_prompt):
                response = self._generate_response(prompt, use_ref=True)
                reward = self._evaluate_math_response(prompt, response)
                original_responses.append(response)
                original_rewards.append(reward)
            
            # Test trained model (current policy)
            trained_responses = []
            trained_rewards = []
            for _ in range(num_responses_per_prompt):
                response = self._generate_response(prompt, use_ref=False)
                reward = self._evaluate_math_response(prompt, response)
                trained_responses.append(response)
                trained_rewards.append(reward)
            
            # Store results
            results['original_model'].append({
                'prompt': prompt,
                'responses': original_responses,
                'rewards': original_rewards,
                'avg_reward': np.mean(original_rewards),
                'max_reward': np.max(original_rewards)
            })
            
            results['trained_model'].append({
                'prompt': prompt,
                'responses': trained_responses,
                'rewards': trained_rewards,
                'avg_reward': np.mean(trained_rewards),
                'max_reward': np.max(trained_rewards)
            })
            
            # Log comparison for this problem
            self.logger.info(f"  Original - Avg: {np.mean(original_rewards):.3f}, Max: {np.max(original_rewards):.3f}")
            self.logger.info(f"  Trained  - Avg: {np.mean(trained_rewards):.3f}, Max: {np.max(trained_rewards):.3f}")
            self.logger.info(f"  Improvement: {np.mean(trained_rewards) - np.mean(original_rewards):.3f}")
        
        # Compute overall statistics
        original_avg_rewards = [item['avg_reward'] for item in results['original_model']]
        trained_avg_rewards = [item['avg_reward'] for item in results['trained_model']]
        
        results['comparison_stats'] = {
            'original_mean_reward': np.mean(original_avg_rewards),
            'trained_mean_reward': np.mean(trained_avg_rewards),
            'improvement': np.mean(trained_avg_rewards) - np.mean(original_avg_rewards),
            'improvement_percentage': ((np.mean(trained_avg_rewards) - np.mean(original_avg_rewards)) / 
                                     np.mean(original_avg_rewards)) * 100,
            'problems_improved': sum(1 for orig, train in zip(original_avg_rewards, trained_avg_rewards) 
                                   if train > orig),
            'total_problems': len(test_prompts)
        }
        
        # Log overall results
        self.logger.info("=" * 60)
        self.logger.info("MODEL COMPARISON RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Original model average reward: {results['comparison_stats']['original_mean_reward']:.3f}")
        self.logger.info(f"Trained model average reward: {results['comparison_stats']['trained_mean_reward']:.3f}")
        self.logger.info(f"Improvement: {results['comparison_stats']['improvement']:.3f}")
        self.logger.info(f"Improvement percentage: {results['comparison_stats']['improvement_percentage']:.1f}%")
        self.logger.info(f"Problems improved: {results['comparison_stats']['problems_improved']}/{results['comparison_stats']['total_problems']}")
        self.logger.info("=" * 60)
        
        return results
    
    def _generate_response(self, prompt: str, use_ref: bool = False) -> str:
        """Generate a single response for a given prompt."""
        # Get the appropriate model
        policy_net = self.ref_policy_net if use_ref else self.policy_net
        
        # Set model mode
        if use_ref:
            policy_net.eval()
        else:
            policy_net.train()
        
        try:
            # Tokenize input
            if hasattr(policy_net, 'tokenizer'):
                tokenizer = policy_net.tokenizer
            else:
                # Fallback to simple generation
                return f"Generated response for: {prompt[:50]}..."
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.config.device)
            
            # Generate response
            with torch.no_grad():
                outputs = policy_net.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            
            return response if response else "No response generated"
            
        except Exception as e:
            self.logger.warning(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _evaluate_math_response(self, prompt: str, response: str) -> float:
        """
        Evaluate mathematical response quality.
        
        This is a simplified evaluation function. In practice, you would use
        more sophisticated mathematical reasoning evaluation.
        """
        reward = 0.0
        
        # Length reward (encourage detailed responses)
        if len(response) > 20:
            reward += 0.1
        
        # Mathematical content indicators
        math_indicators = [
            "=", "+", "-", "*", "/", "solve", "calculate", "answer", "result",
            "step", "first", "then", "next", "therefore", "so", "thus",
            "equation", "formula", "substitute", "simplify"
        ]
        
        math_count = sum(1 for indicator in math_indicators if indicator.lower() in response.lower())
        reward += min(math_count * 0.05, 0.3)
        
        # Step-by-step reasoning reward
        step_indicators = ["step", "first", "second", "third", "then", "next", "finally"]
        step_count = sum(1 for indicator in step_indicators if indicator.lower() in response.lower())
        reward += min(step_count * 0.1, 0.2)
        
        # Number presence reward (mathematical problems should have numbers)
        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        if len(numbers) > 0:
            reward += 0.1
        
        # Answer format reward (look for "answer:" or "=" patterns)
        if "answer:" in response.lower() or "=" in response:
            reward += 0.1
        
        # Coherence reward (check for logical flow)
        if len(response.split()) > 10:  # Substantial response
            reward += 0.1
        
        # Add some randomness to simulate different quality responses
        reward += np.random.normal(0, 0.05)
        
        return max(0.0, min(1.0, reward))  # Clamp between 0 and 1


class GRPOBuffer:
    """Experience buffer for GRPO training."""
    
    def __init__(self, capacity: int, device: str = "cpu"):
        """
        Initialize GRPO buffer.
        
        Args:
            capacity: Buffer capacity
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        
        # Initialize buffers
        self.prompts = []
        self.responses = []
        self.rewards = []
        self.group_indices = []
        
        self.size = 0
        self.ptr = 0
    
    def add(self, 
           prompt: str,
           response: str,
           reward: float,
           group_index: int):
        """Add experience to buffer."""
        if self.size < self.capacity:
            self.prompts.append(prompt)
            self.responses.append(response)
            self.rewards.append(reward)
            self.group_indices.append(group_index)
            self.size += 1
        else:
            self.prompts[self.ptr] = prompt
            self.responses[self.ptr] = response
            self.rewards[self.ptr] = reward
            self.group_indices[self.ptr] = group_index
            self.ptr = (self.ptr + 1) % self.capacity
    
    def get_batch(self, batch_size: int) -> Dict[str, List]:
        """Get a batch of experiences."""
        if self.size < batch_size:
            batch_size = self.size
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'prompts': [self.prompts[i] for i in indices],
            'responses': [self.responses[i] for i in indices],
            'rewards': [self.rewards[i] for i in indices],
            'group_indices': [self.group_indices[i] for i in indices]
        }
    
    def clear(self):
        """Clear the buffer."""
        self.prompts.clear()
        self.responses.clear()
        self.rewards.clear()
        self.group_indices.clear()
        self.size = 0
        self.ptr = 0


def create_simple_lm_head(vocab_size: int, hidden_dim: int = 512) -> nn.Module:
    """Create a simple language model head."""
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, vocab_size),
        nn.LogSoftmax(dim=-1)
    )


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create simple example
    vocab_size = 1000
    hidden_dim = 512
    
    # Create networks
    policy_net = create_simple_lm_head(vocab_size, hidden_dim)
    ref_policy_net = create_simple_lm_head(vocab_size, hidden_dim)
    
    # Create GRPO config
    config = GRPOConfig(
        learning_rate=3e-4,
        clip_ratio=0.2,
        kl_coef=0.1,
        num_responses_per_prompt=4
    )
    
    # Create trainer
    trainer = GRPOTrainer(policy_net, ref_policy_net, config)
    
    # Create buffer
    buffer = GRPOBuffer(capacity=1000)
    
    print("GRPO Algorithm Implementation")
    print("=" * 40)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Number of responses per prompt: {config.num_responses_per_prompt}")
    print(f"Clip ratio: {config.clip_ratio}")
    print(f"KL coefficient: {config.kl_coef}")
    print("GRPO implementation completed!")
