#!/usr/bin/env python3
"""
Mini DPO (Direct Preference Optimization) Training Script
========================================================

This script implements a minimal DPO training pipeline for fine-tuning language models
using preference data. It supports LoRA fine-tuning and various configuration options.

Features:
- Direct Preference Optimization (DPO) algorithm
- LoRA support for memory efficiency
- Configurable training parameters
- Automatic mixed precision training
- Progress tracking and evaluation

Usage:
    python mini_dpo.py

Configuration:
    The script uses direct configuration values (no environment variables needed):
    - Base model: Qwen3-4B-Instruct-2507
    - Reference checkpoint: /home/awpc/studies/models/unsloth/Qwen3/FTTrained
    - Training data: dataset/processed_datasets/qwen3_sft_mixed/train.jsonl
    - Evaluation data: dataset/processed_datasets/qwen3_sft_mixed/validation.jsonl
    - Batch size: 1, Gradient accumulation: 16
    - Learning rate: 8e-5, Epochs: 1
    - Max length: 1024, Beta: 0.3
    - LoRA enabled if PEFT is available
"""

import os
import sys
import json
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Add project root to path
current_path = Path(__file__).resolve()
project_root = current_path.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA training will be disabled.")

# =========================
# Configuration
# =========================
def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/dpo_training.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

def setup_wandb(config: Dict, logger: logging.Logger):
    """Setup Weights & Biases logging."""
    if not config.get("use_wandb", False):
        logger.info("Wandb logging disabled")
        return None
    
    try:
        # Initialize wandb
        wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            entity=config.get("wandb_entity"),
            config={
                "model": config["base_model"],
                "ref_model": config["ref_ckpt"],
                "batch_size": config["batch_size"],
                "grad_accum": config["grad_accum"],
                "learning_rate": config["learning_rate"],
                "epochs": config["epochs"],
                "max_length": config["max_length"],
                "beta": config["beta"],
                "use_lora": config["use_lora"],
                "use_quantization": config.get("use_quantization", False),
                "device": config["device"],
                "dtype": str(config["dtype"]),
            },
            tags=["dpo", "qwen3", "lora", "preference-optimization"]
        )
        logger.info(f"Wandb initialized: {wandb.run.url}")
        return wandb
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        return None

def get_config():
    """Get configuration with direct values (no environment variables)."""
    config = {
        # Model configuration - use Qwen 0.6B
        "base_model": "Qwen/Qwen3-0.6B",
        "ref_ckpt": "Qwen/Qwen3-0.6B",  # Use same model as reference
        "train_dataset": "dataset/combined_pairs/train",
        "eval_dataset": "dataset/combined_pairs/eval",
        
        # Training configuration - match SFT config
        "batch_size": 1,  # Reduce batch size to save memory
        "grad_accum": 24,  # Increase gradient accumulation to maintain effective batch size
        "learning_rate": 1e-5,  # Match SFT learning_rate
        "epochs": 1,  # Match SFT num_train_epochs
        "max_length": 1024,  # Reduce max length to save memory
        "beta": 0.1,  # DPO beta parameter
        "length_norm": True,
        "use_lora": PEFT_AVAILABLE,
        
        # Hardware configuration - match SFT config
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        "use_quantization": False,  # 0.6B model doesn't need quantization
        
        # Output configuration
        "output_dir": "../../../models/transformers/Qwen3-0.6B/DPOTrained",  
        "save_steps": 100,
        "eval_steps": 50,
        
        # Weights & Biases configuration
        "use_wandb": True,
        "wandb_project": "qwen3-dpo-training",
        "wandb_run_name": "qwen3-4b-dpo-lora",
        "wandb_entity": None,  # Set to your wandb username if needed
    }
    
    return config

# Load configuration
logger = setup_logging()
CONFIG = get_config()

# Create output directory if it doesn't exist
os.makedirs(CONFIG["output_dir"], exist_ok=True)
logger.info(f"Output directory: {CONFIG['output_dir']}")


# =========================
# Data Processing
# =========================
@dataclass
class Batch:
    pos_input_ids: torch.Tensor
    pos_labels:    torch.Tensor
    pos_mask:      torch.Tensor
    neg_input_ids: torch.Tensor
    neg_labels:    torch.Tensor
    neg_mask:      torch.Tensor

def build_chat_prompt(tokenizer, prompt: str, response: str) -> Tuple[List[int], int]:
    """
    Build chat prompt with the same chat template as SFT.
    Align labels: only calculate loss in the response segment (prompt masked).
    
    Args:
        tokenizer: The tokenizer to use
        prompt: User prompt
        response: Assistant response
        
    Returns:
        Tuple of (token_ids, prefix_length)
    """
    # For Qwen/Llama, use official chat_template
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "user", "content": prompt}, 
            {"role": "assistant", "content": response}
        ]
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        
        # Get prefix for label masking
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}],
            tokenize=False, 
            add_generation_prompt=False
        )
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    else:
        # Fallback template: <s>User: ... \nAssistant: ...</s>
        prefix_text = f"User: {prompt}\nAssistant:"
        text = prefix_text + " " + response
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        prefix_ids = tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
    
    return ids, len(prefix_ids)

def collate_fn(examples: List[Dict], tokenizer, max_length: int) -> Batch:
    """
    Collate function for DPO training batches.
    
    Args:
        examples: List of preference examples
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Batch object with positive and negative examples
    """
    def make_side(resp_key: str):
        """Process one side (chosen/rejected) of the preference data."""
        seqs, prefix_lens = [], []
        
        for ex in examples:
            ids, pref_len = build_chat_prompt(tokenizer, ex["prompt"], ex[resp_key])
            # Truncate to max_length
            seqs.append(ids[:max_length])
            prefix_lens.append(min(pref_len, max_length))
        
        # Find actual max length in this batch
        actual_max_len = min(max_length, max(len(s) for s in seqs))
        
        input_ids = []
        labels = []
        label_mask = []
        
        for ids, pref_len in zip(seqs, prefix_lens):
            # Pad sequences
            pad_len = actual_max_len - len(ids)
            arr = ids + [tokenizer.pad_token_id] * pad_len
            input_ids.append(arr)
            
            # Labels for next-token prediction (shifted input_ids)
            lab = arr.copy()
            
            # Mask: only response tokens participate in loss
            # prefix position=0; pad=0; response=1
            response_len = max(0, len(ids) - pref_len)
            m = [0] * pref_len + [1] * response_len + [0] * pad_len
            labels.append(lab)
            label_mask.append(m)
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(label_mask, dtype=torch.float32)
        )

    # Process both chosen and rejected responses
    pos_input_ids, pos_labels, pos_mask = make_side("chosen")
    neg_input_ids, neg_labels, neg_mask = make_side("rejected")
    
    return Batch(pos_input_ids, pos_labels, pos_mask, neg_input_ids, neg_labels, neg_mask)


# =========================
# DPO Loss and Helper Functions
# =========================
def seq_logprob_from_logits(
    logits: torch.Tensor, 
    labels: torch.Tensor, 
    mask: torch.Tensor, 
    length_norm: bool
) -> torch.Tensor:
    """
    Compute sequence log probabilities from logits.
    
    Args:
        logits: [B, T, V] - model logits
        labels: [B, T] - target token ids
        mask: [B, T] - mask for response tokens only
        length_norm: whether to normalize by length
        
    Returns:
        [B] - sequence log probabilities
    """
    log_prob = F.log_softmax(logits, dim=-1)
    tok_log_prob = log_prob.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
    tok_log_prob = tok_log_prob * mask
    
    if length_norm: # logprob is divided by length ∣𝑦∣， i.e. get average logprob
        denom = mask.sum(-1).clamp_min(1.0)
        return tok_log_prob.sum(-1) / denom
    else:
        return tok_log_prob.sum(-1)

@torch.no_grad()
def forward_ref(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Forward pass through reference model (frozen)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits

def forward_policy(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Forward pass through policy model (trainable)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits

def dpo_loss(
    pol_pos_lp: torch.Tensor, 
    pol_neg_lp: torch.Tensor,
    ref_pos_lp: torch.Tensor, 
    ref_neg_lp: torch.Tensor,
    beta: float
) -> torch.Tensor:
    """
    Compute DPO loss.
    
    Args:
        pol_pos_lp: Policy model log prob for positive examples
        pol_neg_lp: Policy model log prob for negative examples
        ref_pos_lp: Reference model log prob for positive examples
        ref_neg_lp: Reference model log prob for negative examples
        beta: DPO temperature parameter
        
    Returns:
        DPO loss scalar
    """
    # Δ = (logπθ(y+)-logπθ(y-)) - (logπref(y+)-logπref(y-))
    delta = (pol_pos_lp - pol_neg_lp) - (ref_pos_lp - ref_neg_lp)
    # -log σ(βΔ) = softplus(-βΔ)
    return F.softplus(-beta * delta).mean()


# =========================
# Main Training Function
# =========================
def load_models(config: Dict, logger: logging.Logger):
    """Load policy and reference models."""
    logger.info("Loading models...")
    
    # Clear GPU memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU memory cache")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    # Load policy model (trainable) with optional quantization
    logger.info(f"Loading policy model from: {config['base_model']}")
    
    # Configure quantization if enabled
    quantization_config = None
    if config.get("use_quantization", False):
        from transformers import BitsAndBytesConfig
        logger.info("Configuring 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", # nf4 is better than q4_0
            bnb_4bit_compute_dtype=config["dtype"],
            bnb_4bit_use_double_quant=True,
        )
    
    policy = AutoModelForCausalLM.from_pretrained(
        config["base_model"], 
        torch_dtype=config["dtype"],
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto" if config.get("use_quantization", False) else None
    )
    
    if not config.get("use_quantization", False):
        policy = policy.to(config["device"])
    
    # Apply LoRA if enabled - match SFT config
    if config["use_lora"]:
        logger.info("Applying LoRA configuration...")
        lora_cfg = LoraConfig(
            r=32,  # Match SFT LORA_CONFIG["r"]
            lora_alpha=64,  # Match SFT LORA_CONFIG["lora_alpha"]
            lora_dropout=0.01,  # Match SFT LORA_CONFIG["lora_dropout"]
            bias="none",  # Match SFT LORA_CONFIG["bias"]
            task_type="CAUSAL_LM",  # Match SFT LORA_CONFIG["task_type"]
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]  # Match SFT LORA_CONFIG["target_modules"]
        )
        policy = get_peft_model(policy, lora_cfg)
        policy.print_trainable_parameters()
    
    # Load reference model (frozen) - use transformers only
    logger.info(f"Loading reference model from: {config['ref_ckpt']}")
    
    # Configure quantization for reference model if enabled
    ref_quantization_config = None
    if config.get("use_quantization", False):
        from transformers import BitsAndBytesConfig
        ref_quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=config["dtype"],
            bnb_4bit_use_double_quant=True,
        )
    
    ref = AutoModelForCausalLM.from_pretrained(
        config["ref_ckpt"], 
        torch_dtype=config["dtype"],
        quantization_config=ref_quantization_config,
        trust_remote_code=True,
        device_map="auto" if config.get("use_quantization", False) else None
    )
    
    if not config.get("use_quantization", False):
        ref = ref.to(config["device"])
    
    # Freeze reference model
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    
    logger.info("Models loaded successfully!")
    return tokenizer, policy, ref

def train_epoch(policy, ref, train_loader, tokenizer, optimizer, scheduler, scaler, config, logger, epoch, wandb_logger=None):
    """Train for one epoch."""
    policy.train()
    total_loss = 0
    step = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        pos_ids = batch.pos_input_ids.to(config["device"])
        neg_ids = batch.neg_input_ids.to(config["device"])
        pos_mask_tok = batch.pos_mask.to(config["device"])
        neg_mask_tok = batch.neg_mask.to(config["device"])
        
        # Create attention masks
        pos_attn = (pos_ids != tokenizer.pad_token_id).long()
        neg_attn = (neg_ids != tokenizer.pad_token_id).long()
        
        # Use modern autocast API with fallback for older PyTorch versions
        with torch.amp.autocast(device_type="cuda", dtype=config["dtype"], enabled=True):
            # Policy model forward passes
            pol_pos_logits = forward_policy(policy, pos_ids, pos_attn)
            pol_neg_logits = forward_policy(policy, neg_ids, neg_attn)
            
            # Reference model forward passes (frozen)
            with torch.no_grad():
                ref_pos_logits = forward_ref(ref, pos_ids, pos_attn)
                ref_neg_logits = forward_ref(ref, neg_ids, neg_attn)
            
            # Compute log probabilities (shifted for next-token prediction)
            # pol_pos_logits[:, :-1]   # shape [B, T-1, V]
            # pos_ids[:, 1:]           # shape [B, T-1]
            # pos_mask_tok[:, 1:]      # shape [B, T-1]
            pol_pos_lp = seq_logprob_from_logits(
                pol_pos_logits[:, :-1], pos_ids[:, 1:], pos_mask_tok[:, 1:], config["length_norm"]
            )
            pol_neg_lp = seq_logprob_from_logits(
                pol_neg_logits[:, :-1], neg_ids[:, 1:], neg_mask_tok[:, 1:], config["length_norm"]
            )
            ref_pos_lp = seq_logprob_from_logits(
                ref_pos_logits[:, :-1], pos_ids[:, 1:], pos_mask_tok[:, 1:], config["length_norm"]
            )
            ref_neg_lp = seq_logprob_from_logits(
                ref_neg_logits[:, :-1], neg_ids[:, 1:], neg_mask_tok[:, 1:], config["length_norm"]
            )
            
            # Compute DPO loss
            loss = dpo_loss(pol_pos_lp, pol_neg_lp, ref_pos_lp, ref_neg_lp, config["beta"]) / config["grad_accum"]
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights
        if (batch_idx + 1) % config["grad_accum"] == 0:
            # Calculate gradient norm before clipping and stepping
            grad_norm = 0.0
            if hasattr(torch.nn.utils, 'clip_grad_norm_'):
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0).item()
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            
            # Log progress
            if step % 10 == 0:
                current_loss = loss.item() * config['grad_accum']
                logger.info(f"Epoch {epoch}, Step {step} | Loss: {current_loss:.4f} | Grad Norm: {grad_norm:.4f}")
                
                # Log to wandb
                if wandb_logger:
                    wandb_logger.log({
                        "train/loss": current_loss,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/step": step,
                        "train/grad_norm": grad_norm
                    })
        
        total_loss += loss.item() * config["grad_accum"]
    
    return total_loss / len(train_loader)

def evaluate(policy, ref, eval_loader, tokenizer, config, logger, wandb_logger=None):
    """Evaluate the model on validation set."""
    policy.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            # Move to device
            pos_ids = batch.pos_input_ids.to(config["device"])
            neg_ids = batch.neg_input_ids.to(config["device"])
            pos_mask_tok = batch.pos_mask.to(config["device"])
            neg_mask_tok = batch.neg_mask.to(config["device"])
            
            # Create attention masks
            pos_attn = (pos_ids != tokenizer.pad_token_id).long()
            neg_attn = (neg_ids != tokenizer.pad_token_id).long()
            
            # Use modern autocast API with fallback for older PyTorch versions
            try:
                with torch.amp.autocast(device_type="cuda", dtype=config["dtype"], enabled=True):
                    # Forward passes
                    pol_pos_logits = forward_policy(policy, pos_ids, pos_attn)
                    pol_neg_logits = forward_policy(policy, neg_ids, neg_attn)
                    ref_pos_logits = forward_ref(ref, pos_ids, pos_attn)
                    ref_neg_logits = forward_ref(ref, neg_ids, neg_attn)
                    
                    # Compute log probabilities
                    pol_pos_lp = seq_logprob_from_logits(
                        pol_pos_logits[:, :-1], pos_ids[:, 1:], pos_mask_tok[:, 1:], config["length_norm"]
                    )
                    pol_neg_lp = seq_logprob_from_logits(
                        pol_neg_logits[:, :-1], neg_ids[:, 1:], neg_mask_tok[:, 1:], config["length_norm"]
                    )
                    ref_pos_lp = seq_logprob_from_logits(
                        ref_pos_logits[:, :-1], pos_ids[:, 1:], pos_mask_tok[:, 1:], config["length_norm"]
                    )
                    ref_neg_lp = seq_logprob_from_logits(
                        ref_neg_logits[:, :-1], neg_ids[:, 1:], neg_mask_tok[:, 1:], config["length_norm"]
                    )
                    
                    # Compute DPO loss
                    loss = dpo_loss(pol_pos_lp, pol_neg_lp, ref_pos_lp, ref_neg_lp, config["beta"])
                    total_loss += loss.item()
            except Exception as e:
                print(f"Exception: {e}")
    
    avg_loss = total_loss / len(eval_loader)
    
    # Log evaluation metrics to wandb
    if wandb_logger:
        wandb_logger.log({
            "eval/loss": avg_loss,
            "eval/epoch": getattr(evaluate, 'current_epoch', 0)
        })
    
    return avg_loss

def main():
    """Main training function."""
    logger.info("Starting DPO training...")
    
    # Create output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Setup wandb logging
    wandb_logger = setup_wandb(CONFIG, logger)
    
    # Load models
    tokenizer, policy, ref = load_models(CONFIG, logger)
    
    # Load datasets
    logger.info("Loading datasets...")
    from datasets import load_from_disk
    
    # Load from Arrow files directly
    train_ds = load_from_disk(CONFIG["train_dataset"])
    eval_ds = load_from_disk(CONFIG["eval_dataset"])
    logger.info(f"Loaded train dataset: {len(train_ds)} samples")
    logger.info(f"Loaded eval dataset: {len(eval_ds)} samples")
    
    # Validate dataset format
    if len(train_ds) > 0:
        sample = train_ds[0]
        logger.info(f"Sample data keys: {list(sample.keys())}")
        if not all(key in sample for key in ["prompt", "chosen", "rejected"]):
            logger.error("Dataset missing required fields: prompt, chosen, rejected")
            sys.exit(1)
    
    # Create data loaders
    collator = lambda batch: collate_fn(batch, tokenizer, CONFIG["max_length"])
    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        collate_fn=collator
    )
    eval_loader = DataLoader(
        eval_ds, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        collate_fn=collator
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(policy.parameters(), lr=CONFIG["learning_rate"])
    total_steps = (len(train_loader) * CONFIG["epochs"]) // CONFIG["grad_accum"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=max(10, total_steps // 20), 
        num_training_steps=total_steps
    )
    
    # Setup mixed precision scaler
    try:
        # Modern PyTorch (>= 2.0)
        scaler = torch.amp.GradScaler(device="cuda", enabled=(CONFIG["dtype"] == torch.float16))
    except AttributeError:
        # Fallback for older PyTorch versions
        scaler = torch.cuda.amp.GradScaler(enabled=(CONFIG["dtype"] == torch.float16))
    
    logger.info(f"Training configuration:")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Batch size: {CONFIG['batch_size']}")
    logger.info(f"  Gradient accumulation: {CONFIG['grad_accum']}")
    logger.info(f"  Learning rate: {CONFIG['learning_rate']}")
    logger.info(f"  Beta: {CONFIG['beta']}")
    
    # Training loop
    for epoch in range(CONFIG["epochs"]):
        logger.info(f"Starting epoch {epoch + 1}/{CONFIG['epochs']}")
        
        # Train
        train_loss = train_epoch(policy, ref, train_loader, tokenizer, optimizer, scheduler, scaler, CONFIG, logger, epoch + 1, wandb_logger)
        logger.info(f"Epoch {epoch + 1} train loss: {train_loss:.4f}")
        
        # Evaluate
        eval_loss = evaluate(policy, ref, eval_loader, tokenizer, CONFIG, logger, wandb_logger)
        logger.info(f"Epoch {epoch + 1} eval loss: {eval_loss:.4f}")
        
        # Log epoch-level metrics to wandb
        if wandb_logger:
            wandb_logger.log({
                "epoch": epoch + 1,
                "epoch/train_loss": train_loss,
                "epoch/eval_loss": eval_loss,
                "epoch/learning_rate": scheduler.get_last_lr()[0]
            })
    
    # Save final model
    output_path = os.path.join(CONFIG["output_dir"], "dpo_policy_lora" if CONFIG["use_lora"] else "dpo_policy_full")
    logger.info(f"Saving model to: {output_path}")
    policy.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Log final metrics to wandb
    if wandb_logger:
        wandb_logger.log({
            "final/train_loss": train_loss,
            "final/eval_loss": eval_loss,
            "final/model_path": output_path
        })
        wandb_logger.finish()
    
    logger.info("DPO training completed successfully!")

if __name__ == "__main__":
    main()
