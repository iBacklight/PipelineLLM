"""
GRPO Training Script for Qwen3-0.6B with GSM8K Dataset and LoRA

This script demonstrates how to train Qwen3-0.6B using GRPO algorithm
with GSM8K mathematical reasoning dataset and LoRA fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import json
import os
from tqdm import tqdm
import wandb
from datetime import datetime
import re

from grpo_algorithm import GRPOConfig, GRPOTrainer


@dataclass
class GRPOQwenConfig:
    """Configuration for GRPO training with Qwen3-0.6B."""
    
    # Model configuration
    model_name: str = "Qwen/Qwen-0.6B"  # Using Qwen2.5-0.5B as closest available
    max_length: int = 256  # Reduced from 512 to save memory
    device_map: str = "auto"
    
    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Quantization configuration
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    
    # GRPO configuration
    learning_rate: float = 5e-5
    clip_ratio: float = 0.2
    kl_coef: float = 0.1
    num_responses_per_prompt: int = 2  # Reduced from 4 to save memory
    batch_size: int = 1  # Reduced from 2 to save memory
    num_epochs: int = 20
    
    # Training configuration
    max_train_samples: int = 1000
    max_eval_samples: int = 200
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10
    
    # Wandb configuration
    use_wandb: bool = True
    wandb_project: str = "grpo-qwen3-gsm8k"
    wandb_entity: Optional[str] = None
    wandb_run_name: str = "grpo-qwen3-0-gsm8k"
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class Qwen3GRPOTrainer:
    """GRPO Trainer for Qwen3-0.6B with LoRA and GSM8K dataset."""
    
    def __init__(self, config: GRPOQwenConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Load dataset
        self._load_dataset()
        
        # Create GRPO trainer
        self._create_grpo_trainer()
        
        # Initialize wandb
        self._init_wandb()
        
    def _load_model_and_tokenizer(self):
        """Load Qwen3 model and tokenizer with quantization and LoRA."""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        # Configure quantization
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map=self.config.device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Create reference model (frozen copy)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map=self.config.device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.logger.info("Model and tokenizer loaded successfully")
        self.logger.info(f"Trainable parameters: {self.model.num_parameters()}")
    
    def _load_dataset(self):
        """Load and prepare GSM8K dataset."""
        self.logger.info("Loading GSM8K dataset")
        
        # Load GSM8K dataset
        dataset = load_dataset("gsm8k", "main")
        
        # Prepare training data
        self.train_data = []
        for i, example in enumerate(dataset["train"]):
            if i >= self.config.max_train_samples:
                break
            
            # Format the prompt with structured output format
            prompt = f"""Question: {example['question']}

Please solve this step by step and provide your final answer in the format "#### [number]".

Answer:"""
            response = f" {example['answer']}"
            
            # Extract ground truth from the answer column
            ground_truth = self._extract_final_answer_from_solution(example['answer'])
            
            self.train_data.append({
                "prompt": prompt,
                "response": response,
                "question": example['question'],
                "answer": example['answer'],
                "ground_truth": ground_truth
            })
        
        # Prepare evaluation data
        self.eval_data = []
        for i, example in enumerate(dataset["test"]):
            if i >= self.config.max_eval_samples:
                break
            
            prompt = f"""Question: {example['question']}

Please solve this step by step and provide your final answer in the format "#### [number]".

Answer:"""
            response = f" {example['answer']}"
            
            # Extract ground truth from the answer column
            ground_truth = self._extract_final_answer_from_solution(example['answer'])
            
            self.eval_data.append({
                "prompt": prompt,
                "response": response,
                "question": example['question'],
                "answer": example['answer'],
                "ground_truth": ground_truth
            })
        
        self.logger.info(f"Loaded {len(self.train_data)} training samples")
        self.logger.info(f"Loaded {len(self.eval_data)} evaluation samples")
    
    def _create_grpo_trainer(self):
        """Create GRPO trainer with Qwen3 models."""
        grpo_config = GRPOConfig(
            learning_rate=self.config.learning_rate,
            clip_ratio=self.config.clip_ratio,
            kl_coef=self.config.kl_coef,
            num_responses_per_prompt=self.config.num_responses_per_prompt,
            batch_size=self.config.batch_size,
            device=str(self.device)
        )
        
        self.grpo_trainer = GRPOTrainer(
            policy_net=self.model,
            ref_policy_net=self.ref_model,
            config=grpo_config
        )
        
        # Add tokenizer to the GRPO trainer
        self.grpo_trainer.tokenizer = self.tokenizer
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        if self.config.use_wandb:
            # Generate run name if not provided
            if self.config.wandb_run_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.config.wandb_run_name = f"grpo-qwen3-{timestamp}"
            
            # Initialize wandb
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.wandb_run_name,
                config={
                    "model_name": self.config.model_name,
                    "max_length": self.config.max_length,
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                    "lora_dropout": self.config.lora_dropout,
                    "learning_rate": self.config.learning_rate,
                    "clip_ratio": self.config.clip_ratio,
                    "kl_coef": self.config.kl_coef,
                    "num_responses_per_prompt": self.config.num_responses_per_prompt,
                    "batch_size": self.config.batch_size,
                    "num_epochs": self.config.num_epochs,
                    "max_train_samples": self.config.max_train_samples,
                    "max_eval_samples": self.config.max_eval_samples,
                    "load_in_4bit": self.config.load_in_4bit,
                    "bnb_4bit_quant_type": self.config.bnb_4bit_quant_type,
                },
                tags=["grpo", "qwen3", "gsm8k", "lora", "rlhf"]
            )
            
            # Log model architecture info
            wandb.watch(self.model, log="gradients", log_freq=100)
            
            self.logger.info(f"Wandb initialized: {self.config.wandb_project}/{self.config.wandb_run_name}")
        else:
            self.logger.info("Wandb logging disabled")
    
    def generate_responses(self, prompts: List[str], num_responses: int = None) -> List[Tuple[str, str, float]]:
        """
        Generate multiple responses for each prompt using Qwen3.
        
        Args:
            prompts: List of input prompts
            num_responses: Number of responses per prompt
            
        Returns:
            List of (prompt, response, reward) tuples
        """
        if num_responses is None:
            num_responses = self.config.num_responses_per_prompt
        
        responses_data = []
        
        for prompt in tqdm(prompts, desc="Generating responses"):
            # Find ground truth for this prompt
            ground_truth = None
            for item in self.train_data + self.eval_data:
                if item["prompt"] == prompt:
                    ground_truth = item["ground_truth"]
                    break
            
            for _ in range(num_responses):
                # Generate response using Qwen3
                response = self._generate_single_response(prompt)
                
                # Compute reward using GSM8K evaluation with ground truth
                reward = self._compute_gsm8k_reward(prompt, response, ground_truth)
                
                responses_data.append((prompt, response, reward))
        
        return responses_data
    
    def _generate_single_response(self, prompt: str) -> str:
        """Generate a single response using Qwen3."""
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,  # Reduced from 256 to save memory
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        return response
    
    # def _compute_gsm8k_reward(self, prompt: str, response: str) -> float:
    #     """
    #     Compute reward for GSM8K mathematical reasoning.
        
    #     This is a simplified reward function. In practice, you would use
    #     a more sophisticated evaluation method.
    #     """
    #     # Extract the question from the prompt
    #     if "Question:" in prompt:
    #         question = prompt.split("Question:")[1].split("Answer:")[0].strip()
    #     else:
    #         question = prompt
        
    #     # Simple reward based on response length and mathematical content
    #     reward = 0.0
        
    #     # Length reward (encourage detailed responses)
    #     if len(response) > 50:
    #         reward += 0.2
        
    #     # Mathematical content reward
    #     math_indicators = ["=", "+", "-", "*", "/", "solve", "calculate", "answer"]
    #     math_count = sum(1 for indicator in math_indicators if indicator in response.lower())
    #     reward += min(math_count * 0.1, 0.5)
        
    #     # Step-by-step reasoning reward
    #     if "step" in response.lower() or "first" in response.lower() or "then" in response.lower():
    #         reward += 0.3
        
    #     # Add some randomness to simulate different quality responses
    #     reward += np.random.normal(0, 0.1)
        
    #     return max(0.0, min(1.0, reward))  # Clamp between 0 and 1

    def _extract_final_answer_from_solution(self, solution: str) -> float:
        """
        Extract the final answer from the solution using multiple patterns.
        
        Tries to find the answer in various formats:
        - "Final Answer: [number]"
        - "#### [number]" (GSM8K format)
        - "Answer: [number]"
        - "The answer is [number]"
        
        Args:
            solution: the solution string
        Returns:
            the final answer as float, or None if not found
        """
        # Try multiple patterns in order of preference
        # Prioritize #### format since that's what GSM8K uses
        patterns = [
            r"####\s*([-+]?[0-9]*\.?[0-9]+)",          # "#### 42" (GSM8K format)
            r"Final Answer:\s*([-+]?[0-9]*\.?[0-9]+)",  # "Final Answer: 42"
            r"Answer:\s*([-+]?[0-9]*\.?[0-9]+)",       # "Answer: 42"
            r"The answer is\s*([-+]?[0-9]*\.?[0-9]+)", # "The answer is 42"
            r"=?\s*([-+]?[0-9]*\.?[0-9]+)\s*$",        # "= 42" at end of line
        ]
        
        for pattern in patterns:
            m = re.search(pattern, solution.strip(), re.IGNORECASE)
            if m:
                try:
                    ans = float(m.group(1))
                    # if the number is an integer, convert it to int
                    if ans.is_integer():
                        ans = int(ans)
                    return ans
                except ValueError:
                    continue
        
        return None


    def _compute_gsm8k_reward(self, prompt: str, response: str, ground_truth: float = None,
                            length_bonus_threshold: int = 50,
                            math_indicator_weight: float = 0.1,
                            max_math_indicator_bonus: float = 0.5,
                            step_word_bonus: float = 0.3,
                            length_bonus: float = 0.2,
                            noise_std: float = 0.1) -> float:
        """
        Compute reward for GSM8K mathematical reasoning using hybrid approach.
        
        Combines correctness evaluation (when answer can be extracted) with
        response quality indicators for robust reward computation.
        
        Args:
            prompt: the question text (string)
            response: the model's answer and reasoning text (string)
            ground_truth: the official correct answer value (optional)
            length_bonus_threshold: if the solution length > this threshold, give length_bonus score
            math_indicator_weight: the weight of each mathematical identifier
            max_math_indicator_bonus: the maximum bonus for mathematical identifiers
            step_word_bonus: if the solution contains step keywords, give step_word_bonus score
            length_bonus: if the solution length > threshold, give length_bonus score
            noise_std: the standard deviation of the random noise
        Returns:
            reward: float, between 0.0 and 1.0
        """
        reward = 0.0

        # Try to extract final answer and check correctness
        final_answer = self._extract_final_answer_from_solution(response)
        if final_answer is not None and ground_truth is not None:
            # We have both extracted answer and ground truth - check correctness
            if abs(float(final_answer) - float(ground_truth)) < 1e-6:
                reward += 0.6  # Correct answer gets high base reward
            else:
                reward += 0.1  # Wrong answer gets lower base reward
        else:
            # Fallback to quality-based reward
            reward += 0.0  # Base reward for attempting to solve

        # Length bonus (encourage detailed responses)
        if len(response) > length_bonus_threshold:
            reward += length_bonus

        # Mathematical identifier bonus (encourage mathematical reasoning)
        math_indicators = ["=", "+", "-", "*", "/", "solve", "calculate", "answer", 
                          "therefore", "so", "hence", "multiply", "divide", "add", "subtract"]
        math_count = sum(1 for indicator in math_indicators if indicator in response.lower())
        bonus_math = min(math_count * math_indicator_weight, max_math_indicator_bonus)
        reward += bonus_math

        # Reasoning step keyword bonus (encourage structured thinking)
        step_keywords = ["step", "first", "then", "next", "finally", "second", "third", "now", "so"]
        if any(k in response.lower() for k in step_keywords):
            reward += step_word_bonus


        # Format compliance bonus (encourage following instructions)
        # Prioritize #### format since that's what GSM8K uses
        if "####" in response:
            reward += 0.15  # Higher bonus for GSM8K format
        elif "Final Answer:" in response:
            reward += 0.1   # Lower bonus for other formats

        # Add random noise to simulate response quality variation
        reward += np.random.normal(0.0, noise_std)

        # Clamp to [0.0, 1.0]
        reward = max(0.0, min(1.0, reward))

        return reward
    
    def train(self):
        """Train the model using GRPO algorithm."""
        self.logger.info("Starting GRPO training")
        
        # Prepare training prompts
        train_prompts = [item["prompt"] for item in self.train_data]
        
        # Training metrics tracking
        epoch_rewards = []
        epoch_losses = []
        global_step = 0
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Generate responses for a batch of prompts
            batch_prompts = train_prompts[:self.config.batch_size]
            responses_data = self.generate_responses(batch_prompts)
            
            # Group responses by prompt
            group_indices = self._group_responses_by_prompt(responses_data)
            
            # Extract data for GRPO update
            prompts = [item[0] for item in responses_data]
            responses = [item[1] for item in responses_data]
            rewards = [item[2] for item in responses_data]
            
            # Update policy using GRPO
            stats = self.grpo_trainer.update(prompts, responses, rewards, group_indices)
            
            # Track metrics
            epoch_rewards.extend(rewards)
            epoch_losses.append(stats.get('total_loss', 0))
            
            # Clear cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log training statistics
            self.logger.info(f"Epoch {epoch + 1} - Total Loss: {stats.get('total_loss', 0):.4f}")
            self.logger.info(f"Epoch {epoch + 1} - Policy Loss: {stats.get('policy_loss', 0):.4f}")
            self.logger.info(f"Epoch {epoch + 1} - KL Penalty: {stats.get('kl_penalty', 0):.4f}")
            self.logger.info(f"Epoch {epoch + 1} - Avg Reward: {np.mean(rewards):.4f}")
            self.logger.info(f"Epoch {epoch + 1} - Max Reward: {np.max(rewards):.4f}")
            self.logger.info(f"Epoch {epoch + 1} - Min Reward: {np.min(rewards):.4f}")
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/total_loss": stats.get('total_loss', 0),
                    "train/policy_loss": stats.get('policy_loss', 0),
                    "train/kl_penalty": stats.get('kl_penalty', 0),
                    "train/kl_mean": stats.get('kl_mean', 0),
                    "train/kl_std": stats.get('kl_std', 0),
                    "train/ratio_mean": stats.get('ratio_mean', 0),
                    "train/ratio_std": stats.get('ratio_std', 0),
                    "train/advantages_mean": stats.get('advantages_mean', 0),
                    "train/advantages_std": stats.get('advantages_std', 0),
                    "rewards/mean": np.mean(rewards),
                    "rewards/std": np.std(rewards),
                    "rewards/max": np.max(rewards),
                    "rewards/min": np.min(rewards),
                    "rewards/median": np.median(rewards),
                    "rewards/q25": np.percentile(rewards, 25),
                    "rewards/q75": np.percentile(rewards, 75),
                    "group_stats/num_groups": len(set(group_indices)),
                    "group_stats/avg_group_size": len(responses) / len(set(group_indices)),
                }, step=global_step)
                
                # Log sample responses every few epochs
                if (epoch + 1) % 2 == 0:
                    self.log_sample_responses(prompts, responses, rewards, global_step)
            
            global_step += 1
            
            # Evaluate periodically
            if (epoch + 1) % 2 == 0:
                eval_metrics = self.evaluate()
                
                # Log evaluation metrics to wandb
                if self.config.use_wandb and eval_metrics:
                    wandb.log({
                        "eval/avg_reward": eval_metrics.get('avg_reward', 0),
                        "eval/avg_reward_std": eval_metrics.get('avg_reward_std', 0),
                        "eval/avg_group_reward": eval_metrics.get('avg_group_reward', 0),
                        "eval/num_groups": eval_metrics.get('num_groups', 0),
                        "eval/total_responses": eval_metrics.get('total_responses', 0),
                    }, step=global_step)
        
        # Log final epoch statistics
        if self.config.use_wandb:
            wandb.log({
                "final/avg_epoch_reward": np.mean(epoch_rewards),
                "final/std_epoch_reward": np.std(epoch_rewards),
                "final/avg_epoch_loss": np.mean(epoch_losses),
                "final/std_epoch_loss": np.std(epoch_losses),
            })
        
        self.logger.info("Training completed!")
        
        # Finish wandb run
        if self.config.use_wandb:
            wandb.finish()
    
    def _group_responses_by_prompt(self, responses_data: List[Tuple[str, str, float]]) -> List[int]:
        """Group responses by their originating prompt."""
        prompt_to_group = {}
        group_indices = []
        current_group = 0
        
        for prompt, response, reward in responses_data:
            if prompt not in prompt_to_group:
                prompt_to_group[prompt] = current_group
                current_group += 1
            
            group_indices.append(prompt_to_group[prompt])
        
        return group_indices
    
    def evaluate(self):
        """Evaluate the model on GSM8K test set."""
        self.logger.info("Evaluating model")
        
        eval_prompts = [item["prompt"] for item in self.eval_data[:10]]  # Evaluate on 10 samples
        
        eval_rewards = []
        eval_responses = []
        
        for prompt in eval_prompts:
            # Find ground truth for this prompt
            ground_truth = None
            for item in self.eval_data:
                if item["prompt"] == prompt:
                    ground_truth = item["ground_truth"]
                    break
            
            # Generate response
            response = self._generate_single_response(prompt)
            
            # Compute reward with ground truth
            reward = self._compute_gsm8k_reward(prompt, response, ground_truth)
            eval_rewards.append(reward)
            eval_responses.append(response)
            
            self.logger.info(f"Prompt: {prompt[:100]}...")
            self.logger.info(f"Response: {response[:100]}...")
            self.logger.info(f"Reward: {reward:.3f}")
            self.logger.info("-" * 50)
        
        # Calculate evaluation metrics
        avg_reward = np.mean(eval_rewards) if eval_rewards else 0.0
        std_reward = np.std(eval_rewards) if eval_rewards else 0.0
        
        # Group-level analysis
        group_rewards = []
        for i in range(0, len(eval_rewards), self.config.num_responses_per_prompt):
            group = eval_rewards[i:i + self.config.num_responses_per_prompt]
            if group:
                group_rewards.append(np.mean(group))
        
        avg_group_reward = np.mean(group_rewards) if group_rewards else 0.0
        
        # Response quality metrics
        response_lengths = [len(resp.split()) for resp in eval_responses]
        avg_response_length = np.mean(response_lengths) if response_lengths else 0.0
        
        # Mathematical content analysis
        math_indicators = ["=", "+", "-", "*", "/", "solve", "calculate", "answer", "step"]
        math_scores = []
        for resp in eval_responses:
            math_count = sum(1 for indicator in math_indicators if indicator in resp.lower())
            math_scores.append(math_count)
        
        avg_math_score = np.mean(math_scores) if math_scores else 0.0
        
        # Compile metrics
        metrics = {
            'avg_reward': avg_reward,
            'avg_reward_std': std_reward,
            'avg_group_reward': avg_group_reward,
            'num_groups': len(group_rewards),
            'total_responses': len(eval_rewards),
            'avg_response_length': avg_response_length,
            'avg_math_score': avg_math_score,
            'max_reward': np.max(eval_rewards) if eval_rewards else 0.0,
            'min_reward': np.min(eval_rewards) if eval_rewards else 0.0,
            'median_reward': np.median(eval_rewards) if eval_rewards else 0.0,
        }
        
        self.logger.info(f"Average evaluation reward: {avg_reward:.3f} ± {std_reward:.3f}")
        self.logger.info(f"Average group reward: {avg_group_reward:.3f}")
        self.logger.info(f"Number of groups: {len(group_rewards)}")
        self.logger.info(f"Average response length: {avg_response_length:.1f} words")
        self.logger.info(f"Average math score: {avg_math_score:.1f}")
        
        return metrics
    
    def log_sample_responses(self, prompts: List[str], responses: List[str], rewards: List[float], step: int):
        """Log sample responses to wandb for qualitative analysis."""
        if not self.config.use_wandb:
            return
        
        # Create a table of sample responses
        table_data = []
        for i, (prompt, response, reward) in enumerate(zip(prompts[:5], responses[:5], rewards[:5])):
            table_data.append([
                f"Sample {i+1}",
                prompt[:100] + "..." if len(prompt) > 100 else prompt,
                response[:100] + "..." if len(response) > 100 else response,
                f"{reward:.3f}"
            ])
        
        # Log as wandb table
        table = wandb.Table(
            columns=["Sample", "Prompt", "Response", "Reward"],
            data=table_data
        )
        
        wandb.log({
            "sample_responses": table,
            "sample_rewards_histogram": wandb.Histogram(rewards),
        }, step=step)
    
    def save_model(self, output_dir: str):
        """Save the trained model."""
        self.logger.info(f"Saving model to {output_dir}")
        
        # Save LoRA weights
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training config
        config_dict = {
            "model_name": self.config.model_name,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "learning_rate": self.config.learning_rate,
            "clip_ratio": self.config.clip_ratio,
            "kl_coef": self.config.kl_coef,
        }
        
        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Log model artifacts to wandb
        # if self.config.use_wandb:
        #     # Create model artifact
        #     model_artifact = wandb.Artifact(
        #         name=f"grpo-qwen3-model-{self.config.wandb_run_name}",
        #         type="model",
        #         description=f"GRPO trained Qwen3 model with LoRA on GSM8K dataset"
        #     )
            
        #     # Add model files to artifact
        #     model_artifact.add_dir(output_dir)
            
            # # Log the artifact
            # wandb.log_artifact(model_artifact)
            
            # # Log model summary
            # wandb.summary.update({
            #     "model_saved": True,
            #     "model_path": output_dir,
            #     "final_config": config_dict
            # })
        
        self.logger.info("Model saved successfully")


def main():
    """Main training function."""
    # Create configuration
    config = GRPOQwenConfig(
        model_name="Qwen/Qwen3-0.6B",
        max_train_samples=100,
        max_eval_samples=20,
        batch_size=2,
        num_epochs=20,
        num_responses_per_prompt=4,
        use_wandb=True,
        wandb_project="grpo-qwen3-gsm8k",
        wandb_run_name="grpo-qwen3-0.6b-experiment"
    )
    
    # Create trainer
    trainer = Qwen3GRPOTrainer(config)
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model("/home/awpc/studies/models/transformers/Qwen3-0.6B/GRPOTrained/grpo_qwen3_gsm8k_model")
    
    print("GRPO training completed!")


if __name__ == "__main__":
    main()
