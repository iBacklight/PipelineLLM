#!/usr/bin/env python3
"""
Model Merging Script for VLLM Deployment
========================================

This script merges LoRA adapters with the base model and prepares them for VLLM deployment.
It also creates a backup copy of the original base model.

Usage:
    python merge_model.py --adapter_path /path/to/adapter --base_model_path /path/to/base --output_path /path/to/output
"""

import os
import sys
import argparse
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import unsloth
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# Add project root to path
current_path = Path(__file__).resolve()
project_root = current_path.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import get_model_config, get_output_config


class ModelMerger:
    """Class to handle model merging operations."""
    
    def __init__(self, log_level: str = "INFO"):
        """Initialize the model merger."""
        self.logger = self._setup_logging(log_level)
        self.model_config = get_model_config()
        self.output_config = get_output_config()
        
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('model_merging.log', mode='a')
            ]
        )
        return logging.getLogger(__name__)
    
    def copy_base_model(self, base_model_path: str, backup_path: str) -> bool:
        """
        Create a backup copy of the original base model.
        
        Args:
            base_model_path: Path to the original base model
            backup_path: Path where to save the backup
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info(f"Creating backup copy of base model...")
        self.logger.info(f"Source: {base_model_path}")
        self.logger.info(f"Backup: {backup_path}")
        
        try:
            # Create backup directory
            os.makedirs(backup_path, exist_ok=True)
            
            # Copy all files from base model to backup
            if os.path.isdir(base_model_path):
                shutil.copytree(base_model_path, backup_path, dirs_exist_ok=True)
                self.logger.info(f"Base model backup created successfully at {backup_path}")
            else:
                self.logger.error(f"Base model path is not a directory: {base_model_path}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create base model backup: {e}")
            return False
    
    def load_model_and_tokenizer(self, model_path: str):
        """Load model and tokenizer."""
        self.logger.info(f"Loading model from: {model_path}")
        
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
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_path,
                **model_args
            )
            
            # Setup pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                self.logger.info("Set pad_token to eos_token")
            
            self.logger.info("Model and tokenizer loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def merge_models(self, 
                    base_model_path: str, 
                    adapter_path: str, 
                    output_path: str,
                    save_method: str = "lora",
                    max_shard_size: str = "5GB") -> bool:
        """
        Merge LoRA adapter with base model using merge_and_unload().
        
        Args:
            base_model_path: Path to the base model
            adapter_path: Path to the LoRA adapter
            output_path: Path to save the merged model
            save_method: Method to use for saving ("lora", "4bit", "16bit", "8bit")
            max_shard_size: Maximum size per model shard
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Starting model merging process...")
        self.logger.info(f"Base model: {base_model_path}")
        self.logger.info(f"Adapter: {adapter_path}")
        self.logger.info(f"Output: {output_path}")
        self.logger.info(f"Save method: {save_method}")
        
        try:
            # Load the base model
            self.logger.info("Loading base model...")
            base_model, tokenizer = self.load_model_and_tokenizer(base_model_path)
            
            # Load the LoRA adapter
            self.logger.info("Loading LoRA adapter...")
            adapter_model, _ = self.load_model_and_tokenizer(adapter_path)
            
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Method 1: Use merge_and_unload() for proper LoRA merging
            self.logger.info("Merging LoRA adapter using merge_and_unload()...")
            
            # Check if the model has LoRA adapters
            if hasattr(adapter_model, 'merge_and_unload'):
                # Merge and unload LoRA weights into the base model
                merged_model = adapter_model.merge_and_unload()
                self.logger.info("LoRA adapter merged successfully using merge_and_unload()")
            else:
                # Fallback to Unsloth's method
                self.logger.warning("merge_and_unload() not available, using Unsloth method...")
                FastLanguageModel.save_pretrained_merged(
                    model=adapter_model,
                    tokenizer=tokenizer,
                    save_directory=output_path,
                    save_method=save_method,
                    max_shard_size=max_shard_size,
                )
                # Also save tokenizer separately for compatibility
                tokenizer.save_pretrained(f"{output_path}/tokenizer")
                self.logger.info(f"Merged model saved successfully to {output_path}")
                return True
            
            # Save the merged model using transformers
            self.logger.info("Saving merged model...")
            merged_model.save_pretrained(
                output_path,
                max_shard_size=max_shard_size,
                safe_serialization=True  # Use safetensors format
            )
            
            # Save tokenizer
            tokenizer.save_pretrained(output_path)
            
            # Create model info file
            model_info = {
                "model_type": "merged_lora",
                "base_model": base_model_path,
                "adapter_model": adapter_path,
                "merge_method": "merge_and_unload",
                "merged_at": str(Path().cwd()),
                "note": "This model was merged using merge_and_unload() method"
            }
            
            import json
            info_path = os.path.join(output_path, "merge_info.json")
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            self.logger.info(f"Merged model saved successfully to {output_path}")
            self.logger.info(f"Model can now be loaded with: AutoModelForCausalLM.from_pretrained('{output_path}')")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to merge models: {e}")
            return False
    
    def prepare_for_vllm(self, merged_model_path: str, vllm_output_path: str) -> bool:
        """
        Prepare merged model for VLLM deployment.
        
        Args:
            merged_model_path: Path to the merged model
            vllm_output_path: Path to save VLLM-ready model
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Preparing model for VLLM deployment...")
        
        try:
            # Create VLLM output directory
            os.makedirs(vllm_output_path, exist_ok=True)
            
            # Copy merged model files to VLLM directory
            if os.path.isdir(merged_model_path):
                # Copy all model files
                for item in os.listdir(merged_model_path):
                    src = os.path.join(merged_model_path, item)
                    dst = os.path.join(vllm_output_path, item)
                    
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
                
                # Create VLLM-specific files
                self._create_vllm_config(vllm_output_path)
                
                self.logger.info(f"✅ VLLM-ready model prepared at {vllm_output_path}")
                return True
            else:
                self.logger.error(f"❌ Merged model path is not a directory: {merged_model_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare model for VLLM: {e}")
            return False
    
    def _create_vllm_config(self, vllm_path: str):
        """Create VLLM-specific configuration files."""
        # Create a simple VLLM config
        vllm_config = {
            "model_name": "qwen3-4b-merged",
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": 1,
        }
        
        import json
        config_path = os.path.join(vllm_path, "vllm_config.json")
        with open(config_path, 'w') as f:
            json.dump(vllm_config, f, indent=2)
        
        self.logger.info(f"Created VLLM config at {config_path}")
    
    def run_full_merge_pipeline(self, 
                               base_model_path: str,
                               adapter_path: str,
                               output_path: str,
                               backup_path: Optional[str] = None,
                               save_method: str = "lora",
                               prepare_vllm: bool = True) -> bool:
        """
        Run the complete model merging pipeline.
        
        Args:
            base_model_path: Path to the base model
            adapter_path: Path to the LoRA adapter
            output_path: Path to save the merged model
            backup_path: Path to save base model backup (optional)
            save_method: Method to use for saving
            prepare_vllm: Whether to prepare for VLLM deployment
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("🚀 Starting full model merging pipeline...")
        
        try:
            # Step 1: Create backup of base model (if requested)
            if backup_path:
                if not self.copy_base_model(base_model_path, backup_path):
                    return False
            
            # Step 2: Merge models
            if not self.merge_models(base_model_path, adapter_path, output_path, save_method):
                return False
            
            # Step 3: Prepare for VLLM (if requested)
            if prepare_vllm:
                vllm_path = f"{output_path}_vllm"
                if not self.prepare_for_vllm(output_path, vllm_path):
                    return False
            
            self.logger.info("🎉 Model merging pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model merging pipeline failed: {e}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model for VLLM deployment")
    
    # Required arguments
    parser.add_argument("--base_model_path", required=True, help="Path to the base model")
    parser.add_argument("--adapter_path", required=True, help="Path to the LoRA adapter")
    parser.add_argument("--output_path", required=True, help="Path to save the merged model")
    
    # Optional arguments
    parser.add_argument("--backup_path", help="Path to save base model backup")
    parser.add_argument("--save_method", default="lora", choices=["lora", "4bit", "16bit", "8bit"],
                       help="Method to use for saving merged model")
    parser.add_argument("--max_shard_size", default="5GB", help="Maximum size per model shard")
    parser.add_argument("--prepare_vllm", action="store_true", default=True,
                       help="Prepare model for VLLM deployment")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Create merger instance
    merger = ModelMerger(log_level=args.log_level)
    
    # Run the merging pipeline
    success = merger.run_full_merge_pipeline(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        backup_path=args.backup_path,
        save_method=args.save_method,
        prepare_vllm=args.prepare_vllm
    )
    
    if success:
        print("✅ Model merging completed successfully!")
        sys.exit(0)
    else:
        print("❌ Model merging failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
