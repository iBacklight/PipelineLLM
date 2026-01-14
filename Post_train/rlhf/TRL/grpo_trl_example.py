"""
Simple GRPO Example using TRL (Transformer Reinforcement Learning)

This script demonstrates how to fine-tune a language model using GRPO
(Group Relative Policy Optimization) with the TRL library.

GRPO is a variant of policy optimization that:
1. Generates multiple responses per prompt (a group)
2. Computes relative advantages within each group
3. Uses group-normalized rewards to reduce variance

Key differences from PPO:
- No separate value/critic model needed
- Advantages are computed relative to the group mean
- More sample-efficient for sparse reward settings

Requirements:
    pip install trl transformers torch peft accelerate datasets
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from datasets import Dataset
import re


def create_math_dataset(num_samples=50):
    """
    Create a simple math problem dataset for demonstration.
    GRPO works well with verifiable tasks like math.
    """
    import random
    
    data = []
    for _ in range(num_samples):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        op = random.choice(['+', '-', '*'])
        
        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        else:
            answer = a * b
        
        prompt = f"Calculate: {a} {op} {b} = ?"
        data.append({
            "prompt": prompt,
            "answer": str(answer),
        })
    
    return Dataset.from_list(data)


def math_reward_function(completions, prompts, answers):
    """
    Reward function for math problems.
    Returns 1.0 if the answer is correct, 0.0 otherwise.
    
    This is a verifiable reward - the key advantage of GRPO.
    """
    rewards = []
    for completion, answer in zip(completions, answers):
        # Extract numbers from the completion
        numbers = re.findall(r'-?\d+', completion)
        
        # Check if the correct answer appears in the response
        if answer in numbers:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards


def main():
    # =========================================
    # 1. Configuration
    # =========================================
    model_name = "Qwen/Qwen3-0.6B"  # Use a small model for demo
    
    # GRPO Configuration
    grpo_config = GRPOConfig(
        output_dir="./grpo_output",
        
        # Training parameters
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        
        # GRPO specific parameters
        num_generations=4,               # Number of responses per prompt (group size)
        max_completion_length=64,        # Max length of generated responses
        
        # KL divergence penalty
        beta=0.1,                        # KL penalty coefficient
        
        # Logging
        logging_steps=10,
        save_steps=100,
        
        # Generation parameters
        temperature=0.7,
        
        # Optimization
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        
        # Use bf16 for efficiency
        bf16=True,
        
        # Seed for reproducibility
        seed=42,
    )
    
    # =========================================
    # 2. Load Model and Tokenizer
    # =========================================
    print("Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # =========================================
    # 3. Apply LoRA (optional but recommended)
    # =========================================
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # =========================================
    # 4. Create Dataset
    # =========================================
    print("Creating math dataset...")
    dataset = create_math_dataset(num_samples=100)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample: {dataset[0]}")
    
    # =========================================
    # 5. Define Reward Function
    # =========================================
    def reward_fn(completions, prompts, **kwargs):
        """
        Wrapper reward function for GRPO trainer.
        
        Args:
            completions: List of generated text completions
            prompts: List of input prompts
            **kwargs: Additional metadata (e.g., answers)
        
        Returns:
            List of reward values
        """
        # In practice, you might use a reward model here
        # For math, we use exact answer matching
        
        rewards = []
        for completion, prompt in zip(completions, prompts):
            # Parse the expected answer from the prompt
            # Format: "Calculate: a op b = ?"
            match = re.search(r'(\d+)\s*([+\-*])\s*(\d+)', prompt)
            if match:
                a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
                if op == '+':
                    expected = a + b
                elif op == '-':
                    expected = a - b
                else:
                    expected = a * b
                
                # Check if answer is in completion
                if str(expected) in completion:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        
        return rewards
    
    # =========================================
    # 6. Initialize GRPO Trainer
    # =========================================
    print("Initializing GRPO Trainer...")
    
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,         # Custom reward function
        peft_config=lora_config,        # Apply LoRA
    )
    
    # =========================================
    # 7. Train
    # =========================================
    print("\nStarting GRPO training...")
    print("=" * 50)
    print("GRPO Training Process:")
    print("1. For each prompt, generate N responses (group)")
    print("2. Compute rewards for all responses")
    print("3. Normalize rewards within each group")
    print("4. Update policy using group-relative advantages")
    print("=" * 50)
    
    trainer.train()
    
    # =========================================
    # 8. Save the Model
    # =========================================
    print("\nSaving model...")
    trainer.save_model("./grpo_trained_model")
    
    # =========================================
    # 9. Test the Model
    # =========================================
    print("\nTesting trained model...")
    
    test_prompts = [
        "Calculate: 15 + 27 = ?",
        "Calculate: 42 - 18 = ?",
        "Calculate: 7 * 8 = ?",
    ]
    
    model.eval()
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.1,  # Low temperature for deterministic output
                do_sample=True,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")
    
    print("GRPO training complete!")


if __name__ == "__main__":
    main()

