#!/usr/bin/env python3
"""
Training configuration file for Qwen3-4B SFT
Adjust these parameters according to your needs and hardware capabilities
"""

# ==== Model Configuration ====
MODEL_CONFIG = {
    "max_seq_length": 2048,
    "dtype": "bfloat16",
    "load_in_4bit": True,
    # "quant_type": "nf4",  # Removed - not supported by all models
}

# ==== LoRA Configuration ====
LORA_CONFIG = {
    "r": 16,                # LoRA rank - higher values = more parameters but better adaptation
    "lora_alpha": 32,       # LoRA scaling factor - typically 2 * r
    "lora_dropout": 0.01,   # Dropout for regularization
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "up_proj", "down_proj", "gate_proj"       # MLP layers
    ],
}

# ==== Training Configuration ====
TRAINING_CONFIG = {
    # Training duration
    "num_train_epochs": 1,
    # "max_steps": 30,                      # Use max_steps instead of num_train_epochs for control over training duration
    
    # Batch size settings
    "per_device_train_batch_size": 2,        # Adjust based on GPU memory
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,         # Effective batch size = batch_size * accumulation_steps
    
    # Optimization settings
    "optim": "adamw_8bit",
    "learning_rate": 2e-4,                    # Learning rate - typical range: 1e-5 to 5e-4
    "lr_scheduler_type": "cosine",            # Learning rate scheduler
    "warmup_ratio": 0.1,                     # Warmup portion of training
    "weight_decay": 0.01,                    # L2 regularization
    "max_grad_norm": 1.0,                    # Gradient clipping
    
    # Memory optimization
    "gradient_checkpointing": True,           # Trade computation for memory
    "dataloader_pin_memory": False,
    
    # Mixed precision
    "fp16": False,                           # Use fp16 for older GPUs
    "bf16": True,                            # Use bf16 for newer GPUs (RTX 30xx+, A100, etc.)
    
    # Evaluation and saving
    "eval_strategy": "steps",                # Use newer parameter name
    "eval_steps": 100,                       # Evaluate every N steps
    "save_strategy": "steps",
    "save_steps": 500,                       # Save checkpoint every N steps
    "save_total_limit": 3,                   # Keep only last N checkpoints
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    
    # Logging
    "logging_steps": 10,
    "report_to": "wandb",                     # Change to "wandb" or "tensorboard" if you want logging
    
    # Reproducibility
    "seed": 42,
}

# ==== SFT-Specific Configuration ====
# SFTConfig provides additional SFT-specific parameters
SFT_CONFIG = {
    # ==== Training duration ====
    "num_train_epochs": 2,
    # "max_steps": 30,                      # Use max_steps instead of num_train_epochs for precise control
    
    # ==== Batch size settings ====
    "per_device_train_batch_size": 4,        # Adjust based on GPU memory
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,        # Effective batch size = batch_size * accumulation_steps
    
    # ==== Optimization settings ====
    "optim": "adamw_8bit",
    "learning_rate": 2e-4,                   # Typical range: 1e-5 to 5e-4
    "lr_scheduler_type": "cosine",           # Cosine LR scheduler
    "warmup_ratio": 0.1,                     # Warmup portion of training
    "weight_decay": 0.01,                    # L2 regularization
    "max_grad_norm": 1.0,                    # Gradient clipping
    
    # ==== Memory optimization ====
    "gradient_checkpointing": True,          # Trade computation for memory
    "dataloader_pin_memory": False,
    
    # ==== Mixed precision ====
    "fp16": False,                           # Use fp16 for older GPUs
    "bf16": True,                            # Use bf16 for newer GPUs (RTX 30xx+, A100, etc.)
    
    # ==== Evaluation and saving ====
    "eval_strategy": "steps",          # Standard param name in HF
    "eval_steps": 100,                       # Evaluate every N steps
    "save_strategy": "steps",
    "save_steps": 500,                       # Save checkpoint every N steps
    "save_total_limit": 3,                   # Keep only last N checkpoints
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    
    # ==== Logging ====
    "logging_steps": 10,
    "report_to": "wandb",                    # Options: "wandb", "tensorboard", etc.
    
    # ==== Reproducibility ====
    "seed": 42,
    
    # ==== SFT-specific parameters ====
    "packing": True,                         # Pack multiple samples into one sequence
    "dataset_text_field": "text",            # Field name containing the text
    "max_seq_length": 2048,                  # Maximum sequence length
    "neftune_noise_alpha": None,             # NEFTune noise alpha (None = disabled)
    "dataset_num_proc": 4,                   # Number of processes for dataset processing
    "dataset_batch_size": 1000,              # Batch size for dataset processing
    "formatting_func": None,                 # Custom formatting function
    "infinite": False,                       # Use infinite dataset
    "num_of_sequences": 1024,                # Number of sequences when using infinite dataset
    "chars_per_token": 3.6,                  # Avg characters per token for packing
    
    # ==== Loss masking ====
    "assistant_only_loss": True,             # Compute loss only on assistant responses
    "completion_only_loss": None,            # For prompt-completion style datasets
}

# ==== Trainer Configuration ====
TRAINER_CONFIG = {
    "use_sft_config": True,                 # Or True using SFTConfig
    "assistant_only_loss": True,            # Compute loss only on assistant responses (recommended for chat models)
    "sft_specific_args": SFT_CONFIG.copy(), # SFT-specific arguments
}

# ==== Output Configuration ====
OUTPUT_CONFIG = {
    "final_model_dir": "/home/awpc/studies/models/unsloth/Qwen3/FTTrained",
    "logging_dir": "./logs",
    "run_name": "qwen3-4b-sft",
    
    # Checkpoint settings
    "save_checkpoints": True,              # Enable/disable checkpoint saving (saves storage space when False)
    "checkpoint_dir": None,                # None means use final_model_dir for checkpoints (keeps everything together)
    
    # Model saving options
    "save_merged_model": False,             # Save merged model (LoRA + base model)
    "merged_model_save_method": "lora",    # Options: "lora", "4bit", "16bit", "8bit"
    "save_both_formats": False,             # Save both LoRA adapter and merged model
    "max_shard_size": "5GB",               # Maximum size per model shard
}

# ==== Data Configuration ====
DATA_CONFIG = {
    "train_data_path": "dataset/processed_datasets/qwen3_sft_mixed/train",
    "val_data_path": "dataset/processed_datasets/qwen3_sft_mixed/validation",
    "packing": False,                        # Set to True for better efficiency, False for debugging
}

# ==== Hardware Specific Configurations ====
# Choose the appropriate configuration based on your GPU

# For RTX 3090/4090 (24GB VRAM)
GPU_24GB_CONFIG = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 2048,
}

# For RTX 3080/4080 (10-16GB VRAM) 
GPU_16GB_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 1024,
}

# For RTX 3070/4070 (8-12GB VRAM)
GPU_12GB_CONFIG = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "max_seq_length": 1024,
}

# Apply GPU-specific config (uncomment the one that matches your hardware)
# TRAINING_CONFIG.update(GPU_24GB_CONFIG)
# TRAINING_CONFIG.update(GPU_16GB_CONFIG)
# TRAINING_CONFIG.update(GPU_12GB_CONFIG)

# ==== Storage-Saving Configurations ====
# Use these presets to save disk space

# Minimal storage mode - no checkpoints, only final model
MINIMAL_STORAGE_CONFIG = {
    "save_checkpoints": False,
    "save_strategy": "no",
    "evaluation_strategy": "no", 
    "load_best_model_at_end": False,
}

# Checkpoint mode - saves checkpoints with final model (default)
CHECKPOINT_MODE_CONFIG = {
    "save_checkpoints": True,
    "save_strategy": "steps",
    "save_steps": 500,
    "save_total_limit": 3,
    "evaluation_strategy": "steps",
    "eval_steps": 100,
}

# Apply storage configuration (uncomment the one you prefer)
# OUTPUT_CONFIG.update(MINIMAL_STORAGE_CONFIG)    # Save disk space
# OUTPUT_CONFIG.update(CHECKPOINT_MODE_CONFIG)     # Full checkpoint saving (default)

# ==== Helper Functions ====
def get_model_config():
    """Get model configuration"""
    return MODEL_CONFIG.copy()

def get_lora_config():
    """Get LoRA configuration"""
    return LORA_CONFIG.copy()

def get_training_config():
    """Get training configuration"""
    return TRAINING_CONFIG.copy()

def get_sft_config():
    """Get SFT-specific configuration"""
    return SFT_CONFIG.copy()

def get_trainer_config():
    """Get trainer configuration"""
    return TRAINER_CONFIG.copy()

def get_output_config():
    """Get output configuration"""
    return OUTPUT_CONFIG.copy()

def get_data_config():
    """Get data configuration"""
    return DATA_CONFIG.copy()

def get_minimal_storage_config():
    """Get minimal storage configuration (no checkpoints)"""
    return MINIMAL_STORAGE_CONFIG.copy()

def get_checkpoint_mode_config():
    """Get checkpoint mode configuration (full checkpoints)"""
    return CHECKPOINT_MODE_CONFIG.copy()

def enable_minimal_storage():
    """Enable minimal storage mode - only saves final model"""
    OUTPUT_CONFIG.update(MINIMAL_STORAGE_CONFIG)
    TRAINING_CONFIG.update(MINIMAL_STORAGE_CONFIG)
    
def enable_checkpoint_mode():
    """Enable checkpoint mode - saves intermediate checkpoints"""
    OUTPUT_CONFIG.update(CHECKPOINT_MODE_CONFIG)
    TRAINING_CONFIG.update(CHECKPOINT_MODE_CONFIG)

def print_config():
    """Print all configurations"""
    print("="*60)
    print("🔧 TRAINING CONFIGURATION")
    print("="*60)
    
    print("\n📱 Model Configuration:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n🎯 LoRA Configuration:")
    for key, value in LORA_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n🚀 Training Configuration:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n💾 Output Configuration:")
    for key, value in OUTPUT_CONFIG.items():
        if key.startswith("save_") or key.startswith("merged_"):
            print(f"  {key}: {value} {'(Model Saving)' if key in ['save_merged_model', 'save_both_formats'] else ''}")
        else:
            print(f"  {key}: {value}")
    
    print("\n📊 Data Configuration:")
    for key, value in DATA_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n🎯 SFT Configuration:")
    for key, value in SFT_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n⚙️ Trainer Configuration:")
    for key, value in TRAINER_CONFIG.items():
        print(f"  {key}: {value}")
    
    # Calculate effective batch size
    effective_batch_size = (
        TRAINING_CONFIG["per_device_train_batch_size"] * 
        TRAINING_CONFIG["gradient_accumulation_steps"]
    )
    print(f"\n📈 Effective Batch Size: {effective_batch_size}")
    
    # Show checkpoint settings
    checkpoint_status = "✅ ENABLED" if OUTPUT_CONFIG.get("save_checkpoints", True) else "❌ DISABLED (saves storage)"
    checkpoint_dir = OUTPUT_CONFIG.get("checkpoint_dir") or OUTPUT_CONFIG.get("final_model_dir", "./final_model")
    print(f"\n💾 Checkpoint Settings:")
    print(f"  Status: {checkpoint_status}")
    print(f"  Directory: {checkpoint_dir}")
    if OUTPUT_CONFIG.get("save_checkpoints", True):
        save_steps = TRAINING_CONFIG.get("save_steps", "N/A")
        save_limit = TRAINING_CONFIG.get("save_total_limit", "N/A")
        print(f"  Save every: {save_steps} steps")
        print(f"  Keep last: {save_limit} checkpoints")
    else:
        print(f"  Note: Only final model will be saved")
    
    print("="*60)

if __name__ == "__main__":
    print_config()
