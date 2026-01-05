import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import time
from tqdm import tqdm
import json
import os

try:
    from ..utils import (
        Experience,
        ExperienceBuffer,
        compute_advantages_and_returns,
        compute_metrics,
        set_seed,
        pad_sequences,
        create_attention_mask
    )
    from .ppo_algorithm import PPOPolicy
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import (
        Experience,
        ExperienceBuffer,
        compute_advantages_and_returns,
        compute_metrics,
        set_seed,
        pad_sequences,
        create_attention_mask
    )
    from PPO.ppo_algorithm import PPOPolicy


class PPOTrainer:
    """PPO Trainer for RLHF training."""
    
    def __init__(
        self,
        policy: PPOPolicy,
        reward_model: Optional[torch.nn.Module] = None,
        ref_policy: Optional[PPOPolicy] = None,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 32,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_range: float = 0.2,
        value_clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        kl_penalty_coef: float = 0.0,
        kl_target: float = 0.01,
        save_freq: int = 100,
        log_freq: int = 10,
        save_dir: str = "./ppo_checkpoints"
    ):
        self.policy = policy
        self.reward_model = reward_model
        self.ref_policy = ref_policy
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        # PPO hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        
        # KL penalty
        self.kl_penalty_coef = kl_penalty_coef
        self.kl_target = kl_target
        
        # Logging and saving
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer(max_size=10000)
        
        # Training statistics
        self.training_stats = {
            'episode': 0,
            'total_reward': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'kl_divergence': 0.0,
            'clip_ratio': 0.0,
            'mean_reward': 0.0,
            'std_reward': 0.0
        }
    
    def generate_response(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate response using the current policy.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
        
        Returns:
            response_ids: Generated response token IDs
            response_attention_mask: Attention mask for response
            log_probs: Log probabilities of generated tokens
        """
        self.policy.policy_net.eval()
        
        with torch.no_grad():
            batch_size = input_ids.shape[0]
            device = input_ids.device
            
            # Initialize response
            response_ids = torch.empty(batch_size, 0, dtype=torch.long, device=device)
            response_attention_mask = torch.ones(batch_size, 0, dtype=torch.long, device=device)
            log_probs = torch.empty(batch_size, 0, device=device)
            
            # Generate tokens one by one
            for _ in range(max_new_tokens):
                # Concatenate input and current response
                full_input_ids = torch.cat([input_ids, response_ids], dim=1)
                full_attention_mask = torch.cat([attention_mask, response_attention_mask], dim=1)
                
                # Get logits from policy
                logits, _, _ = self.policy.policy_net(full_input_ids, full_attention_mask)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    # Apply top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Get log probability of the chosen token
                next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
                next_token_log_prob = torch.gather(
                    next_token_log_probs, 1, next_token
                ).squeeze(1)
                
                # Update response
                response_ids = torch.cat([response_ids, next_token], dim=1)
                response_attention_mask = torch.cat([
                    response_attention_mask, 
                    torch.ones(batch_size, 1, dtype=torch.long, device=device)
                ], dim=1)
                log_probs = torch.cat([log_probs, next_token_log_prob.unsqueeze(1)], dim=1)
                
                # Check for EOS token (assuming 2 is EOS)
                if (next_token == 2).all():
                    break
        
        return response_ids, response_attention_mask, log_probs
    
    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
        response_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rewards for the generated responses.
        
        Args:
            input_ids: Input token IDs
            response_ids: Generated response token IDs
            response_attention_mask: Attention mask for response
        
        Returns:
            rewards: Computed rewards
        """
        if self.reward_model is not None:
            # Use reward model
            full_input_ids = torch.cat([input_ids, response_ids], dim=1)
            full_attention_mask = torch.cat([
                torch.ones_like(input_ids),
                response_attention_mask
            ], dim=1)
            
            with torch.no_grad():
                rewards = self.reward_model(full_input_ids, full_attention_mask)
        else:
            # Simple reward based on response length (for demonstration)
            # In practice, you would use a trained reward model
            batch_size = response_ids.shape[0]
            rewards = torch.ones(batch_size, device=self.device) * 0.1  # Base reward
            rewards += torch.sum(response_attention_mask, dim=1).float() * 0.01  # Length bonus
        
        return rewards
    
    def compute_kl_penalty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL penalty between current policy and reference policy.
        
        Args:
            input_ids: Input token IDs
            response_ids: Generated response token IDs
            response_attention_mask: Attention mask for response
        
        Returns:
            kl_penalty: KL penalty term
        """
        if self.ref_policy is None or self.kl_penalty_coef == 0.0:
            return torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            # Get log probs from current policy
            full_input_ids = torch.cat([input_ids, response_ids], dim=1)
            full_attention_mask = torch.cat([
                attention_mask,
                response_attention_mask
            ], dim=1)
            
            current_log_probs, _, _ = self.policy.evaluate_actions(
                full_input_ids, full_attention_mask, response_ids
            )
            
            # Get log probs from reference policy
            ref_log_probs, _, _ = self.ref_policy.evaluate_actions(
                full_input_ids, full_attention_mask, response_ids
            )
            
            # Compute KL divergence
            kl_div = (current_log_probs - ref_log_probs).mean()
            kl_penalty = self.kl_penalty_coef * torch.clamp(kl_div - self.kl_target, min=0.0)
        
        return kl_penalty
    
    def collect_experiences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_responses: int = 1
    ) -> List[Experience]:
        """
        Collect experiences by generating responses.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            num_responses: Number of responses to generate per input
        
        Returns:
            experiences: List of collected experiences
        """
        experiences = []
        
        for _ in range(num_responses):
            # Generate response
            response_ids, response_attention_mask, log_probs = self.generate_response(
                input_ids, attention_mask
            )
            
            # Compute rewards
            rewards = self.compute_rewards(input_ids, response_ids, response_attention_mask)
            
            # Compute KL penalty
            kl_penalty = self.compute_kl_penalty(
                input_ids, attention_mask, response_ids, response_attention_mask
            )
            
            # Adjust rewards with KL penalty
            rewards = rewards - kl_penalty
            
            # Get value estimates
            full_input_ids = torch.cat([input_ids, response_ids], dim=1)
            full_attention_mask = torch.cat([
                attention_mask,
                response_attention_mask
            ], dim=1)
            
            with torch.no_grad():
                _, values, _ = self.policy.evaluate_actions(
                    full_input_ids, full_attention_mask, response_ids
                )
            
            # Compute advantages and returns
            advantages, returns = compute_advantages_and_returns(
                rewards, values, self.gamma, self.lam
            )
            
            # Create experience
            experience = Experience(
                input_ids=input_ids,
                attention_mask=attention_mask,
                response_ids=response_ids,
                log_probs=log_probs,
                values=values,
                rewards=rewards,
                advantages=advantages,
                returns=returns
            )
            
            experiences.append(experience)
        
        return experiences
    
    def train_step(self, experiences: List[Experience]) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            experiences: List of experiences to train on
        
        Returns:
            metrics: Training metrics
        """
        # Convert experiences to the format expected by the policy
        experience_dicts = []
        for exp in experiences:
            experience_dicts.append({
                'input_ids': exp.input_ids,
                'attention_mask': exp.attention_mask,
                'action_ids': exp.response_ids,
                'log_probs': exp.log_probs,
                'values': exp.values,
                'advantages': exp.advantages,
                'returns': exp.returns
            })
        
        # Update policy
        metrics = self.policy.update(
            experience_dicts,
            clip_range=self.clip_range,
            value_clip_range=self.value_clip_range,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            max_grad_norm=self.max_grad_norm,
            num_epochs=self.num_epochs
        )
        
        return metrics
    
    def train(
        self,
        train_dataloader,
        num_episodes: int = 1000,
        save_checkpoints: bool = True,
        log_to_file: bool = True
    ):
        """
        Main training loop.
        
        Args:
            train_dataloader: DataLoader for training data
            num_episodes: Number of training episodes
            save_checkpoints: Whether to save model checkpoints
            log_to_file: Whether to log to file
        """
        log_file = os.path.join(self.save_dir, "training.log") if log_to_file else None
        
        for episode in tqdm(range(num_episodes), desc="Training PPO"):
            episode_experiences = []
            episode_rewards = []
            
            # Collect experiences from current batch
            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Collect experiences
                experiences = self.collect_experiences(input_ids, attention_mask)
                episode_experiences.extend(experiences)
                
                # Track rewards
                for exp in experiences:
                    episode_rewards.append(exp.rewards.mean().item())
            
            # Train on collected experiences
            if episode_experiences:
                metrics = self.train_step(episode_experiences)
                
                # Update training statistics
                self.training_stats['episode'] = episode
                self.training_stats['total_reward'] = np.mean(episode_rewards)
                self.training_stats.update(metrics)
                
                # Compute additional metrics
                all_rewards = torch.cat([exp.rewards for exp in episode_experiences])
                all_advantages = torch.cat([exp.advantages for exp in episode_experiences])
                all_returns = torch.cat([exp.returns for exp in episode_experiences])
                
                additional_metrics = compute_metrics(all_rewards, all_advantages, all_returns)
                self.training_stats.update(additional_metrics)
            
            # Logging
            if episode % self.log_freq == 0:
                self._log_metrics(episode, log_file)
            
            # Save checkpoint
            if save_checkpoints and episode % self.save_freq == 0:
                self._save_checkpoint(episode)
    
    def _log_metrics(self, episode: int, log_file: Optional[str] = None):
        """Log training metrics."""
        log_str = f"Episode {episode}: "
        log_str += f"Reward: {self.training_stats['total_reward']:.4f}, "
        log_str += f"Policy Loss: {self.training_stats['policy_loss']:.4f}, "
        log_str += f"Value Loss: {self.training_stats['value_loss']:.4f}, "
        log_str += f"KL Div: {self.training_stats['kl_divergence']:.4f}, "
        log_str += f"Clip Ratio: {self.training_stats['clip_ratio']:.4f}"
        
        print(log_str)
        
        if log_file:
            with open(log_file, 'a') as f:
                f.write(log_str + '\n')
    
    def _save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_episode_{episode}.pt")
        self.policy.save(checkpoint_path)
        
        # Save training stats
        stats_path = os.path.join(self.save_dir, f"stats_episode_{episode}.json")
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
    
    def evaluate(
        self,
        eval_dataloader,
        num_eval_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate the current policy.
        
        Args:
            eval_dataloader: DataLoader for evaluation data
            num_eval_episodes: Number of evaluation episodes
        
        Returns:
            eval_metrics: Evaluation metrics
        """
        self.policy.policy_net.eval()
        
        eval_rewards = []
        eval_metrics = {
            'mean_reward': 0.0,
            'std_reward': 0.0,
            'mean_response_length': 0.0,
            'std_response_length': 0.0
        }
        
        with torch.no_grad():
            for episode in range(num_eval_episodes):
                episode_rewards = []
                episode_lengths = []
                
                for batch in eval_dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # Generate responses
                    response_ids, response_attention_mask, _ = self.generate_response(
                        input_ids, attention_mask, do_sample=False  # Use greedy decoding for evaluation
                    )
                    
                    # Compute rewards
                    rewards = self.compute_rewards(input_ids, response_ids, response_attention_mask)
                    episode_rewards.extend(rewards.cpu().numpy())
                    episode_lengths.extend(response_attention_mask.sum(dim=1).cpu().numpy())
                
                eval_rewards.extend(episode_rewards)
        
        # Compute final metrics
        eval_rewards = np.array(eval_rewards)
        eval_metrics['mean_reward'] = np.mean(eval_rewards)
        eval_metrics['std_reward'] = np.std(eval_rewards)
        eval_metrics['mean_response_length'] = np.mean(episode_lengths)
        eval_metrics['std_response_length'] = np.std(episode_lengths)
        
        return eval_metrics
