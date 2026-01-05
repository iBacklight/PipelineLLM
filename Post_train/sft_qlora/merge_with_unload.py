#!/usr/bin/env python3
"""
LoRA Merge with merge_and_unload() Method
=========================================

This script uses the proper merge_and_unload() method to merge LoRA adapters
with base models, while preserving the original base model as backup.

Usage:
    python merge_with_unload.py --base_model_path /path/to/base --adapter_path /path/to/adapter --output_path /path/to/output
"""

import os
import sys
import argparse
import shutil
import logging
from pathlib import Path
from typing import Optional

def setup_logging(log_level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('merge_unload.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

def copy_base_model(base_model_path: str, backup_path: str, logger) -> bool:
    """
    Create a backup copy of the original base model.
    
    Args:
        base_model_path: Path to the original base model
        backup_path: Path where to save the backup
        logger: Logger instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Creating backup copy of base model...")
    logger.info(f"Source: {base_model_path}")
    logger.info(f"Backup: {backup_path}")
    
    try:
        # Create backup directory
        os.makedirs(backup_path, exist_ok=True)
        
        # Copy all files from base model to backup
        if os.path.isdir(base_model_path):
            shutil.copytree(base_model_path, backup_path, dirs_exist_ok=True)
            logger.info(f"Base model backup created successfully at {backup_path}")
            return True
        else:
            logger.error(f"Base model path is not a directory: {base_model_path}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to create base model backup: {e}")
        return False

def merge_with_unload(base_model_path: str, adapter_path: str, output_path: str, logger) -> bool:
    """
    Merge LoRA adapter with base model using merge_and_unload() method.
    
    Args:
        base_model_path: Path to the base model
        adapter_path: Path to the LoRA adapter
        output_path: Path to save the merged model
        logger: Logger instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting LoRA merge with merge_and_unload()...")
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Adapter: {adapter_path}")
    logger.info(f"Output: {output_path}")
    
    try:
        # Import required libraries
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as e:
            logger.error(f"Required libraries not found: {e}")
            logger.error("Please install: pip install transformers peft")
            return False
        
        # Load base model and tokenizer
        logger.info("Loading base model and tokenizer...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        # Load LoRA adapter
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Merge and unload LoRA weights
        logger.info("Merging LoRA adapter using merge_and_unload()...")
        merged_model = model.merge_and_unload()
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save merged model
        logger.info("Saving merged model...")
        merged_model.save_pretrained(
            output_path,
            max_shard_size="5GB",
            safe_serialization=True  # Use safetensors format
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(output_path)
        
        # Create merge info file
        import json
        model_info = {
            "model_type": "merged_lora",
            "base_model": base_model_path,
            "adapter_model": adapter_path,
            "merge_method": "merge_and_unload",
            "merged_at": str(Path().cwd()),
            "note": "This model was merged using merge_and_unload() method from transformers/peft"
        }
        
        info_path = os.path.join(output_path, "merge_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Merged model saved successfully to {output_path}")
        logger.info(f"Model can now be loaded with: AutoModelForCausalLM.from_pretrained('{output_path}')")
        
        # Clean up memory
        del base_model, model, merged_model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to merge models: {e}")
        return False

def prepare_for_vllm(merged_model_path: str, vllm_output_path: str, logger) -> bool:
    """
    Prepare merged model for VLLM deployment.
    
    Args:
        merged_model_path: Path to the merged model
        vllm_output_path: Path to save VLLM-ready model
        logger: Logger instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Preparing model for VLLM deployment...")
    
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
            vllm_config = {
                "model_name": "qwen3-4b-merged",
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.9,
                "tensor_parallel_size": 1,
            }
            
            import json
            config_path = os.path.join(vllm_output_path, "vllm_config.json")
            with open(config_path, 'w') as f:
                json.dump(vllm_config, f, indent=2)
            
            # Create VLLM deployment script
            vllm_script = '''#!/usr/bin/env python3
"""
VLLM Deployment Script for Merged Model
=======================================

This script demonstrates how to deploy the merged model using VLLM.
"""

from vllm import LLM, SamplingParams

def main():
    # Load the merged model
    llm = LLM(
        model="/path/to/your/merged/model",
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )
    
    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )
    
    # Example prompts
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]
    
    # Generate responses
    outputs = llm.generate(prompts, sampling_params)
    
    # Print results
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Response: {output.outputs[0].text}")
        print("-" * 50)

if __name__ == "__main__":
    main()
'''
            
            script_path = os.path.join(vllm_output_path, "deploy_vllm.py")
            with open(script_path, 'w') as f:
                f.write(vllm_script)
            
            # Make the script executable
            os.chmod(script_path, 0o755)
            
            logger.info(f"VLLM-ready model prepared at {vllm_output_path}")
            logger.info(f"VLLM deployment script created: {script_path}")
            return True
        else:
            logger.error(f"Merged model path is not a directory: {merged_model_path}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to prepare model for VLLM: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model using merge_and_unload()")
    
    # Required arguments
    parser.add_argument("--base_model_path", required=True, help="Path to the base model")
    parser.add_argument("--adapter_path", required=True, help="Path to the LoRA adapter")
    parser.add_argument("--output_path", required=True, help="Path to save the merged model")
    
    # Optional arguments
    parser.add_argument("--backup_path", help="Path to save base model backup")
    parser.add_argument("--prepare_vllm", action="store_true", default=True,
                       help="Prepare model for VLLM deployment")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("Starting LoRA merge with merge_and_unload() pipeline...")
    
    try:
        # Step 1: Create backup of base model (if requested)
        if args.backup_path:
            if not copy_base_model(args.base_model_path, args.backup_path, logger):
                sys.exit(1)
        
        # Step 2: Merge models using merge_and_unload()
        if not merge_with_unload(args.base_model_path, args.adapter_path, args.output_path, logger):
            sys.exit(1)
        
        # Step 3: Prepare for VLLM (if requested)
        if args.prepare_vllm:
            vllm_path = f"{args.output_path}_vllm"
            if not prepare_for_vllm(args.output_path, vllm_path, logger):
                sys.exit(1)
        
        logger.info("LoRA merge pipeline completed successfully!")
        print("\n" + "="*60)
        print("📋 MERGE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✅ Merged model saved to: {args.output_path}")
        if args.backup_path:
            print(f"✅ Base model backup saved to: {args.backup_path}")
        if args.prepare_vllm:
            print(f"✅ VLLM-ready model prepared at: {args.output_path}_vllm")
        print("\n🚀 You can now use the merged model with VLLM or any other inference engine!")
        
    except Exception as e:
        logger.error(f"LoRA merge pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
