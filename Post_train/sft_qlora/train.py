#!/usr/bin/env python3
"""
Qwen3-4B-Instruct Fine-tuning Script
====================================

A comprehensive training script for fine-tuning Qwen3-4B-Instruct model using LoRA/QLoRA.

Author: Your Name
Date: 2024
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import unsloth
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, TrainingArguments
from unsloth import FastLanguageModel
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Add project root to path
current_path = Path(__file__).resolve()
project_root = current_path.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import configuration
from config import (
    get_model_config, get_lora_config, get_training_config,
    get_sft_config, get_trainer_config, get_output_config, 
    get_data_config, print_config
)


class Qwen3Trainer:
    """
    A comprehensive trainer class for fine-tuning Qwen3-4B-Instruct model.
    
    This class encapsulates all the functionality needed for:
    - Model and tokenizer loading
    - Data preprocessing
    - LoRA configuration
    - Training execution
    - Model evaluation and saving
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 config_override: Optional[Dict] = None):
        """
        Initialize the Qwen3Trainer.
        
        Args:
            model_path: Custom path to the model (optional)
            config_override: Dictionary to override default configurations
        """
        self.logger = self._setup_logging()
        self.root_dir = str(project_root)
        
        # Load configurations
        self.model_config = get_model_config()
        self.lora_config = get_lora_config()
        self.training_config = get_training_config()
        self.sft_config = get_sft_config()
        self.trainer_config = get_trainer_config()
        self.output_config = get_output_config()
        self.data_config = get_data_config()
        
        # Apply configuration overrides
        if config_override:
            self._apply_config_override(config_override)
        
        # Set model path
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = self._get_default_model_path()
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.val_dataset = None
        
        self.logger.info("Qwen3Trainer initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('training.log', mode='a')
            ]
        )
        return logging.getLogger(__name__)
    
    def _apply_config_override(self, config_override: Dict):
        """Apply configuration overrides."""
        for key, value in config_override.items():
            if hasattr(self, f"{key}_config"):
                getattr(self, f"{key}_config").update(value)
                self.logger.info(f"Applied override for {key}_config: {value}")
    
    def _get_default_model_path(self) -> str:
        """Get the default model path."""
        return f"{self.root_dir}/models/unsloth/Qwen3/Qwen3-4B-Instruct-2507/models--unsloth--Qwen3-4B-Instruct-2507/snapshots/992063681dc2f7de4ee976110199552935cad284/"
    
    def setup_environment(self):
        """Setup the training environment."""
        self.logger.info("Setting up training environment...")
        
        # Setup checkpoint directory based on configuration
        if self.output_config.get("save_checkpoints", True):
            # Use final_model_dir for checkpoints to keep everything together
            if self.output_config["checkpoint_dir"] is None:
                self.output_config["checkpoint_dir"] = self.output_config["final_model_dir"]
            else:
                self.output_config["checkpoint_dir"] = f"{self.output_config['final_model_dir']}/checkpoints"
            checkpoint_dir = self.output_config["checkpoint_dir"]
            self.output_config["output_dir"] = checkpoint_dir
            self.logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
        else:
            # Use a temporary directory for minimal output when checkpoints are disabled
            self.output_config["output_dir"] = "./temp_training_output"
            self.logger.info("Checkpoint saving is disabled")
        
        # Create output directories
        for dir_key, dir_path in self.output_config.items():
            if dir_key.endswith('_dir') and dir_path:
                os.makedirs(dir_path, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
        
        # Print configuration
        print_config()
        
        # Check CUDA availability
        if torch.cuda.is_available():
            self.logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.logger.warning("CUDA not available, training will be slow on CPU")
    
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        self.logger.info(f"Loading model from: {self.model_path}")
        
        try:
            # Load model with compatible parameters
            model_args = {
                "max_seq_length": self.model_config.get("max_seq_length", 2048),
                "dtype": self.model_config.get("dtype"),
                "load_in_4bit": self.model_config.get("load_in_4bit", True),
                "trust_remote_code": True,
                "device_map": "auto"
            }
            
            # FastLanguageModel.from_pretrained returns (model, tokenizer) tuple
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                self.model_path,
                **model_args
            )
            
            # Setup pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.info("Set pad_token to eos_token")
            
            self.logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def setup_lora(self):
        """Setup LoRA configuration and apply it to the model."""
        self.logger.info("Setting up LoRA configuration...")
        
        try:
            # For Unsloth, we need to pass LoRA parameters directly
            self.model = FastLanguageModel.get_peft_model(
                self.model, 
                r=self.lora_config["r"],                    # LoRA rank
                target_modules=self.lora_config["target_modules"],  # Target modules
                lora_alpha=self.lora_config["lora_alpha"],  # LoRA alpha
                lora_dropout=self.lora_config["lora_dropout"],  # LoRA dropout
                bias=self.lora_config["bias"],              # Bias setting
                use_gradient_checkpointing="unsloth",       # Use Unsloth's checkpointing
                random_state=3407,                          # Random seed
                use_rslora=False,                           # Don't use rank-stabilized LoRA
                loftq_config=None,                          # No LoftQ
            )
            
            # Enable training mode
            self.model = FastLanguageModel.for_training(self.model)
            
            # Log trainable parameters
            self._log_trainable_parameters()
            
            self.logger.info("LoRA configuration applied successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup LoRA: {e}")
            raise
    
    def _log_trainable_parameters(self):
        """Log information about trainable parameters."""
        try:
            trainable_params = 0
            all_params = 0
            
            for _, param in self.model.named_parameters():
                all_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            
            trainable_percent = 100 * trainable_params / all_params if all_params > 0 else 0
            
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            self.logger.info(f"All parameters: {all_params:,}")
            self.logger.info(f"Trainable percentage: {trainable_percent:.2f}%")
        except Exception as e:
            self.logger.warning(f"Could not calculate trainable parameters: {e}")
    
    def load_datasets(self):
        """Load and prepare datasets."""
        self.logger.info("Loading datasets...")
        
        try:
            # Load datasets
            self.train_dataset = load_from_disk(self.data_config["train_data_path"])
            self.val_dataset = load_from_disk(self.data_config["val_data_path"])
            
            self.logger.info(f"Train dataset: {len(self.train_dataset)} samples")
            self.logger.info(f"Validation dataset: {len(self.val_dataset)} samples")
            
            # Format datasets
            self._format_datasets()
            
        except Exception as e:
            self.logger.error(f"Failed to load datasets: {e}")
            raise
    
    def _format_datasets(self):
        """Format datasets using chat templates."""
        self.logger.info("Formatting datasets with chat templates...")
        
        def format_messages(examples):
            """Format messages using Qwen's chat template."""
            texts = []
            for messages in examples["messages"]:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append(text)
            return {"text": texts}
        
        # Apply formatting
        self.train_dataset = self.train_dataset.map(
            format_messages, 
            batched=True, 
            remove_columns=self.train_dataset.column_names
        )
        
        self.val_dataset = self.val_dataset.map(
            format_messages, 
            batched=True, 
            remove_columns=self.val_dataset.column_names
        )
        
        # Log a sample
        sample_text = self.train_dataset[0]['text'][:500]
        self.logger.info(f"Sample formatted text:\n{sample_text}...")
    
    def setup_trainer(self):
        """Setup the SFT trainer."""
        self.logger.info("Setting up trainer...")
        
        try:
            # Calculate effective batch size
            effective_batch_size = (
                self.training_config["per_device_train_batch_size"] * 
                self.training_config["gradient_accumulation_steps"]
            )
            self.logger.info(f"Effective batch size: {effective_batch_size}")
            
            # Modify training config based on checkpoint settings
            modified_training_config = self.training_config.copy()
            if not self.output_config.get("save_checkpoints", True):
                # Disable checkpoint saving to save storage
                modified_training_config["save_strategy"] = "no"
                modified_training_config["eval_strategy"] = "no" 
                modified_training_config["load_best_model_at_end"] = False
                self.logger.info("Checkpoint saving disabled - no intermediate saves will be made")
            
            # Choose between SFTConfig and TrainingArguments
            if self.trainer_config.get("use_sft_config", False):
                self.logger.info("Using SFTConfig for training")
                
                # Combine training config with SFT-specific config
                sft_args = {
                    **modified_training_config,
                    **self.sft_config,
                    "output_dir": self.output_config["output_dir"],
                    "run_name": self.output_config["run_name"],
                    "logging_dir": self.output_config["logging_dir"],
                }
                
                # Log assistant loss masking status
                if self.trainer_config.get("assistant_only_loss", True):
                    self.logger.info("Assistant loss masking ENABLED - loss will be computed only on assistant responses")
                else:
                    self.logger.info("Assistant loss masking DISABLED - loss will be computed on entire sequence")
                
                training_args = SFTConfig(**sft_args)
                
                # Initialize trainer with SFTConfig
                self.trainer = SFTTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.val_dataset,
                    args=training_args,
                )
                
            else:
                self.logger.info("Using TrainingArguments for training")
                
                # Create standard training arguments
                training_args = TrainingArguments(
                    output_dir=self.output_config["output_dir"],
                    run_name=self.output_config["run_name"],
                    logging_dir=self.output_config["logging_dir"],
                    **modified_training_config
                )
                
                # TrainingArguments does not support assistant loss masking
                data_collator = None
                
                # Initialize trainer with TrainingArguments and custom collator
                self.trainer = SFTTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    data_collator=data_collator,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.val_dataset,
                    args=training_args,
                    max_seq_length=self.model_config["max_seq_length"],
                    packing=self.data_config["packing"],
                )
            
            self.logger.info("Trainer setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup trainer: {e}")
            raise
    
    def train(self):
        """Execute the training process."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_trainer() first.")
        
        self.logger.info("Starting training...")
        
        try:
            # Start training
            train_result = self.trainer.train()
            
            # Log training results
            self.logger.info(f"Training completed successfully!")
            self.logger.info(f"Final train loss: {train_result.training_loss:.4f}")
            
            return train_result
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self):
        """Save the trained model and tokenizer."""
        if self.trainer is None:
            raise ValueError("No trained model to save")
        
        self.logger.info(f"Saving model to {self.output_config['final_model_dir']}")
        
        try:
            # Save model
            self.trainer.save_model(self.output_config["final_model_dir"])
            
            # Save tokenizer
            self.tokenizer.save_pretrained(f"{self.output_config['final_model_dir']}/tokenizer")
            
            self.logger.info("Model saved successfully")
            
            # Cleanup temporary directory if checkpoints were disabled
            if not self.output_config.get("save_checkpoints", True):
                temp_dir = self.output_config.get("output_dir")
                if temp_dir and temp_dir.startswith("./temp_") and os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    self.logger.info(f"Cleaned up temporary directory: {temp_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def save_merged_model(self, save_directory: str = None, save_method: str = "lora"):
        """
        Save the merged model using FastLanguageModel.save_pretrained_merged().
        
        This method merges the LoRA adapter with the base model and saves it as a single model.
        This is useful for deployment or when you want to use the model without LoRA.
        
        Args:
            save_directory: Directory to save the merged model. If None, uses final_model_dir + "_merged"
            save_method: Method to use for saving. Options:
                - "lora": Merge LoRA weights (default)
                - "4bit": Save as 4-bit quantized model
                - "16bit": Save as 16-bit model
                - "8bit": Save as 8-bit quantized model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if save_directory is None:
            base_dir = self.output_config["final_model_dir"]
            save_directory = f"{base_dir}_merged"
        
        self.logger.info(f"Saving merged model to {save_directory}")
        self.logger.info(f"Save method: {save_method}")
        
        try:
            # Create save directory
            os.makedirs(save_directory, exist_ok=True)
            
            # Save merged model using Unsloth's method
            FastLanguageModel.save_pretrained_merged(
                model=self.model,
                tokenizer=self.tokenizer,
                save_directory=save_directory,
                save_method=save_method,
                max_shard_size="5GB",  # Split large models into 5GB chunks
            )
            
            # Also save tokenizer separately for compatibility
            self.tokenizer.save_pretrained(f"{save_directory}/tokenizer")
            
            self.logger.info(f"Merged model saved successfully to {save_directory}")
            self.logger.info(f"Model can now be loaded with: AutoModelForCausalLM.from_pretrained('{save_directory}')")
            
        except Exception as e:
            self.logger.error(f"Failed to save merged model: {e}")
            raise
    
    def save_model_both_ways(self, save_method: str = "lora"):
        """
        Save the model in both formats: LoRA adapter and merged model.
        
        Args:
            save_method: Method for merged model saving (lora, 4bit, 16bit, 8bit)
        """
        self.logger.info("Saving model in both formats...")
        
        # Save LoRA adapter (original method)
        self.save_model()
        
        # Save merged model
        self.save_merged_model(save_method=save_method)
        
        self.logger.info("Model saved in both formats successfully!")
    
    def evaluate_model(self, test_prompts: Optional[list] = None):
        """Evaluate the trained model with test prompts."""
        if self.model is None:
            raise ValueError("No model to evaluate")
        
        self.logger.info("Evaluating trained model...")
        
        # Enable inference mode
        FastLanguageModel.for_inference(self.model)
        
        # Default test prompts
        if test_prompts is None:
            test_prompts = [
                "小明有10个苹果，吃了3个，还剩几个？",
                "What is the capital of France?",
                "用Python写一个Hello World程序",
            ]
        
        for i, prompt in enumerate(test_prompts, 1):
            self.logger.info(f"Test {i}: {prompt}")
            
            # Format prompt
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate response
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):], 
                skip_special_tokens=True
            ).strip()
            
            self.logger.info(f"Response: {response}")
    
    def run_full_training_pipeline(self):
        """Run the complete training pipeline."""
        self.logger.info("Starting full training pipeline...")
        
        try:
            # Setup
            self.setup_environment()
            self.load_model_and_tokenizer()
            self.load_datasets()
            self.setup_lora()
            
            self.setup_trainer()
            
            # Train
            train_result = self.train()
            
            # Save model(s)
            if self.output_config.get("save_both_formats", True):
                # Save both LoRA adapter and merged model
                self.save_model_both_ways(
                    save_method=self.output_config.get("merged_model_save_method", "lora")
                )
            elif self.output_config.get("save_merged_model", True):
                # Save only merged model
                self.save_merged_model(
                    save_method=self.output_config.get("merged_model_save_method", "lora")
                )
            else:
                # Save only LoRA adapter (original behavior)
                self.save_model()
            
            # Evaluate
            # self.evaluate_model()
            
            self.logger.info("Training pipeline completed successfully! 🎉")
            return train_result
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main function to run training."""
    try:
        # Initialize trainer
        trainer = Qwen3Trainer()
        
        # Run full training pipeline
        trainer.run_full_training_pipeline()

        print("Training completed successfully!")

        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        logging.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()