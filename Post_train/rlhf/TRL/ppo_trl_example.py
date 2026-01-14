"""
Simple PPO Example using TRL (Transformer Reinforcement Learning)

This script demonstrates how to fine-tune a language model using PPO
with the TRL library from Hugging Face.

PPO (Proximal Policy Optimization) is a popular RL algorithm for RLHF,
which optimizes the policy while constraining updates to stay close to
the original policy using a clipping mechanism.

Requirements:
    pip install trl transformers torch peft accelerate
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model
from datasets import Dataset


def create_dummy_dataset(tokenizer, num_samples=100):
    """
    Create a simple dummy dataset for demonstration.
    In practice, you would load your own prompt dataset.
    """
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "What is the capital of France?",
        "Write a short poem about nature.",
        "How does photosynthesis work?",
        "Describe the water cycle.",
    ] * (num_samples // 5)
    
    # Tokenize the prompts
    tokenized = [tokenizer(p, return_tensors="pt", padding=False, truncation=True, max_length=64) 
                 for p in prompts]
    
    return [{"input_ids": t["input_ids"].squeeze(), "query": p} 
            for t, p in zip(tokenized, prompts)]


def simple_reward_function(responses, prompts):
    """
    A simple reward function for demonstration.
    In practice, you would use a trained reward model.
    
    This dummy function gives higher rewards for:
    - Longer responses (up to a point)
    - Responses that don't repeat too much
    """
    rewards = []
    for response in responses:
        # Base reward based on length (prefer medium-length responses)
        length = len(response)
        length_reward = min(length / 100, 1.0) - max(0, (length - 200) / 200)
        
        # Penalty for repetition
        words = response.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        repetition_reward = unique_ratio
        
        reward = length_reward + repetition_reward
        rewards.append(torch.tensor(reward))
    
    return rewards


def main():
    # =========================================
    # 1. Configuration
    # =========================================
    model_name = "Qwen/Qwen3-0.6B"  # Use a small model for demo
    
    # PPO Configuration
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
        ppo_epochs=2,                    # Number of PPO epochs per batch
        kl_penalty="kl",                 # KL penalty type
        target_kl=0.1,                   # Target KL divergence
        init_kl_coef=0.2,               # Initial KL coefficient
        adap_kl_ctrl=True,              # Adaptive KL control
        clip_range=0.2,                  # PPO clipping range
        vf_coef=0.1,                     # Value function coefficient
        seed=42,
    )
    
    # =========================================
    # 2. Load Model and Tokenizer
    # =========================================
    print("Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with value head for PPO
    # The value head predicts the expected cumulative reward
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Optional: Apply LoRA for parameter-efficient training
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.pretrained_model = get_peft_model(model.pretrained_model, lora_config)
    
    # Load reference model (frozen, for KL penalty computation)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # =========================================
    # 3. Initialize PPO Trainer
    # =========================================
    print("Initializing PPO Trainer...")
    
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    
    # =========================================
    # 4. Create Dataset
    # =========================================
    print("Creating dataset...")
    dataset = create_dummy_dataset(tokenizer, num_samples=20)
    
    # =========================================
    # 5. Training Loop
    # =========================================
    print("Starting PPO training...")
    
    generation_kwargs = {
        "max_new_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    for epoch in range(2):  # Number of epochs
        print(f"\n=== Epoch {epoch + 1} ===")
        
        for batch_idx in range(0, len(dataset), ppo_config.batch_size):
            # Get batch
            batch = dataset[batch_idx:batch_idx + ppo_config.batch_size]
            query_tensors = [item["input_ids"] for item in batch]
            queries = [item["query"] for item in batch]
            
            # Generate responses
            response_tensors = ppo_trainer.generate(
                query_tensors, 
                **generation_kwargs
            )
            
            # Decode responses for reward computation
            responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            
            # Compute rewards (in practice, use a reward model)
            rewards = simple_reward_function(responses, queries)
            
            # PPO step: update the model based on rewards
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # Log statistics
            if batch_idx % 8 == 0:
                print(f"Batch {batch_idx}: "
                      f"reward={stats['ppo/mean_scores']:.3f}, "
                      f"kl={stats['objective/kl']:.3f}")
    
    # =========================================
    # 6. Save the Model
    # =========================================
    print("\nSaving model...")
    model.save_pretrained("./ppo_trained_model")
    tokenizer.save_pretrained("./ppo_trained_model")
    
    print("PPO training complete!")


if __name__ == "__main__":
    main()

