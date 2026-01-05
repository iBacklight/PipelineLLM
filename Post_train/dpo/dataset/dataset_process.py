#!/usr/bin/env python3
"""
Improved DPO Dataset Processing Script
=====================================

This script processes preference datasets for DPO training using built-in dataset library tools.
It creates preference pairs from HelpSteer2 and UltraFeedback-Chinese datasets.

Features:
- Uses built-in dataset operations for deduplication and filtering
- Saves datasets to separate directories
- Simplified and more maintainable code
- Better error handling and logging

Usage:
    python improved_dataset_process.py
"""

import os
import re
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import langid
import json

# Set random seed for reproducibility
random.seed(42)

# ========== Configuration ==========
CONFIG = {
    "datasets": {
        "helpsteer2": "nvidia/HelpSteer2",
        "ultra_cn": "opencsg/UltraFeedback-chinese",
        "ultra_cn_fallback": "HuggingFaceH4/ultrafeedback_binarized"
    },
    "output_dirs": {
        "helpsteer2": "helpsteer2_pairs",
        "ultra_cn": "ultraf_cn_pairs",
        "combined": "combined_pairs"
    },
    "filters": {
        "min_text_length": 16,
        "max_length_delta": 2048,
        "jaccard_threshold": 0.9,
        "test_size": 0.02
    },
    "score_weights": {
        "correctness": 0.4,    # honesty + truthfulness
        "helpfulness": 0.3,    # helpfulness
        "coherence": 0.2,      # instruction_following
        "complexity": 0.05,    # not available in Chinese dataset
        "verbosity": 0.05      # not available in Chinese dataset
    },
    "sampling": {   
        "total_samples": 7500,   # Target total samples
        "en_ratio": 0.7,         # English ratio (70%)
        "zh_ratio": 0.3,         # Chinese ratio (30%)
        "en_samples": 18000,     # English samples (more to ensure enough pairs)
        "en_actual_need_pairs": 5250, # English actual need pairs (70% of 7500)
        "zh_samples": 2250       # Chinese samples (30% of 7500, each sample = 1 pair)
    }
}

# ========== Utility Functions ==========
def is_valid_text(text: str, min_length: int = 16) -> bool:
    """Check if text is valid for training."""
    if not isinstance(text, str) or len(text.strip()) < min_length:
        return False
    
    # Remove PII and control characters
    if re.search(r'(\d{3}-\d{2}-\d{4}|credit card|ssn)', text, re.I):
        return False
    if re.search(r'[\u0000-\u0008\u000B\u000C\u000E-\u001F]', text):
        return False
    
    return True

def detect_language(text: str, target_langs: Tuple[str, ...] = ("en", "zh")) -> bool:
    """Detect if text is in target languages."""
    try:
        lang, _ = langid.classify(text)
        return lang in target_langs
    except Exception:
        return True

def normalize_score(score: Any) -> Optional[float]:
    """Normalize score to [0, 1] range."""
    if score is None:
        return None
    
    try:
        val = float(score)
    except (ValueError, TypeError):
        return None
    
    # Handle different score ranges
    if 0.0 <= val <= 1.0:
        return val
    elif 0.0 <= val <= 5.0:
        return val / 5.0
    elif 0.0 <= val <= 10.0:
        return val / 10.0
    else:
        # Sigmoid compression for other ranges
        try:
            return 1.0 / (1.0 + math.exp(-val))
        except OverflowError:
            return 1.0 if val > 0 else 0.0

def aggregate_scores(scores: Dict[str, Any]) -> Tuple[float, Tuple[float, ...]]:
    """
    Aggregate multiple scores into a single weighted score.
    
    Args:
        scores: Dictionary of scores for different criteria
        
    Returns:
        Tuple of (weighted_sum, tie_breaker_tuple)
    """
    # Normalize individual scores
    norm_scores = {}
    for key, weight in CONFIG["score_weights"].items():
        norm_val = normalize_score(scores.get(key))
        norm_scores[key] = 0.5 if norm_val is None else norm_val
    
    # Calculate weighted sum
    weighted_sum = sum(weight * norm_scores[key] for key, weight in CONFIG["score_weights"].items())
    
    # Create tie-breaker tuple (higher is better)
    tie_breaker = tuple(round(norm_scores[key], 6) for key in CONFIG["score_weights"].keys())
    
    return weighted_sum, tie_breaker

def find_column(candidates: List[str], available_columns: List[str]) -> Optional[str]:
    """Find the first available column from candidates."""
    for candidate in candidates:
        if candidate in available_columns:
            return candidate
    return None

def sample_dataset(dataset: Dataset, target_size: int, dataset_name: str) -> Dataset:
    """Sample dataset to target size with proper logging."""
    current_size = len(dataset)
    print(f"{dataset_name} original size: {current_size}")
    
    if current_size <= target_size:
        print(f"Using all {current_size} samples from {dataset_name}")
        return dataset
    
    # Sample with replacement if needed, otherwise without replacement
    # if target_size > current_size * 0.8:  # If we need more than 80% of data
        # print(f"Sampling {target_size} samples from {dataset_name} (with replacement)")
        # sampled = dataset.shuffle(seed=42).select(range(target_size))
    # else:
    print(f"Sampling {target_size} samples from {dataset_name}")
    sampled = dataset.shuffle(seed=42).select(range(target_size))
    
    print(f"{dataset_name} sampled size: {len(sampled)}")
    
    return sampled

# ========== HelpSteer2 Processing ==========
def process_helpsteer2(dataset_name: str, split: str = "train") -> Dataset:
    """Process HelpSteer2 dataset into preference pairs."""
    print(f"Loading HelpSteer2 dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    
    # Check actual column names
    columns = dataset.column_names
    print(f"HelpSteer2 columns: {columns}")
    
    # HelpSteer2 has: prompt, response, helpfulness, correctness, coherence, complexity, verbosity
    required_cols = ["prompt", "response", "helpfulness", "correctness", "coherence"]
    missing_cols = [col for col in required_cols if col not in columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available: {columns}")
    
    # Sample dataset to target size for English data
    target_size = CONFIG["sampling"]["en_samples"]
    dataset = sample_dataset(dataset, target_size, "HelpSteer2")
    
    def create_preference_pairs(example):
        """Convert single response with scores to preference pairs by comparing with other responses."""
        prompt = example["prompt"]
        response = example["response"]
        
        # Validate text
        if not is_valid_text(prompt) or not is_valid_text(response):
            return None
        
        # Language filtering for English
        if not detect_language(prompt, ("en",)):
            return None
        
        # Collect scores for this response
        scores = {
            "correctness": example.get("correctness", 0),
            "helpfulness": example.get("helpfulness", 0),
            "coherence": example.get("coherence", 0),
            "complexity": example.get("complexity", 0),
            "verbosity": example.get("verbosity", 0)
        }
        
        # Calculate aggregated score
        score, tie_breaker = aggregate_scores(scores)
        
        return {
            "prompt": prompt,
            "response": response,
            "score": score,
            "tie_breaker": tie_breaker,
            "scores": scores
        }
    
    # Process dataset to get scored responses
    scored_responses = dataset.map(create_preference_pairs, remove_columns=columns)
    scored_responses = scored_responses.filter(lambda x: x is not None)
    
    # Process HelpSteer2: adjacent rows are pairs with same prompt but different responses
    all_pairs = []
    scored_responses_list = list(scored_responses)
    
    # Group by prompt (adjacent rows should have same prompt)
    grouped_by_prompt = {}
    for response in scored_responses_list:
        prompt = response["prompt"].strip()
        if prompt not in grouped_by_prompt:
            grouped_by_prompt[prompt] = []
        grouped_by_prompt[prompt].append(response)
    
    # Process each prompt group
    for prompt, responses in grouped_by_prompt.items():
        if len(responses) < 2:
            continue  # Skip if not enough responses for this prompt
        
        # Sort responses by score (descending)
        responses.sort(key=lambda x: (x["score"], x["tie_breaker"]), reverse=True)
        
        # Take the best and worst responses for this prompt
        best = responses[0]  # Highest score
        worst = responses[-1]  # Lowest score
        
        # Skip if responses are too similar
        if best["response"].strip() == worst["response"].strip():
            continue
        
        # Skip if length difference is too large
        if abs(len(best["response"]) - len(worst["response"])) > CONFIG["filters"]["max_length_delta"]:
            continue
        
        all_pairs.append({
            "prompt": prompt,
            "chosen": best["response"],
            "rejected": worst["response"]
        })
    
    result = Dataset.from_list(all_pairs)
    print(f"HelpSteer2 processed: {len(result)} pairs, only return {CONFIG['sampling']['en_actual_need_pairs']} pairs")
    
    # Limit to actual needed pairs
    target_pairs = CONFIG['sampling']['en_actual_need_pairs']
    if len(result) > target_pairs:
        result = result.select(range(target_pairs))
    
    return result

# ========== UltraFeedback-Chinese Processing ==========
def process_ultra_cn(dataset_name: str, split: str = "train") -> Dataset:
    """Process UltraFeedback-Chinese dataset into preference pairs."""
    print(f"Loading UltraFeedback-Chinese dataset: {dataset_name}")
    
    # Calculate how many samples we need (each sample can form 1 pair)
    target_samples = CONFIG["sampling"]["zh_samples"]
    
    try:
        # Try loading with streaming first to avoid schema issues
        print("Loading with streaming to avoid schema issues...")
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        # Take only the samples we need
        dataset = dataset.take(target_samples)
        dataset = Dataset.from_list(list(dataset))
    except Exception as e:
        print(f"Error loading with streaming: {e}")
        try:
            # Try loading with trust_remote_code and limit samples
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
            # Limit to target samples
            if len(dataset) > target_samples:
                dataset = dataset.select(range(target_samples))
        except Exception as e2:
            print(f"Error loading dataset with trust_remote_code: {e2}")
            try:
                # Try loading without trust_remote_code and limit samples
                dataset = load_dataset(dataset_name, split=split)
                # Limit to target samples
                if len(dataset) > target_samples:
                    dataset = dataset.select(range(target_samples))
            except Exception as e3:
                print(f"Error loading dataset: {e3}")
                # Fallback to alternative dataset
                print("Falling back to alternative dataset...")
                dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
                if len(dataset) > target_samples:
                    dataset = dataset.select(range(target_samples))
    
    # Check actual column names
    columns = dataset.column_names
    print(f"UltraFeedback-Chinese columns: {columns}")
    print(f"UltraFeedback-Chinese loaded: {len(dataset)} samples")
    
    # Check for Chinese dataset format: instruction + completions list
    if "instruction" in columns and "completions" in columns:
        return process_chinese_completions_format(dataset, "UltraFeedback-Chinese")
    
    else:
        raise ValueError(f"Unknown dataset format. Available columns: {columns}")

def process_chinese_completions_format(dataset: Dataset, dataset_name: str) -> Dataset:
    """Process Chinese dataset with instruction + completions list format."""
    print(f"Processing {dataset_name} with Chinese completions format")
    
    def extract_scores_from_annotations(annotations):
        """Extract scores from the annotations dictionary."""
        scores = {}
        
        # Map annotation keys to our score keys
        annotation_mapping = {
            "helpfulness": "helpfulness",
            "honesty": "correctness",  # Map honesty to correctness
            "instruction_following": "coherence",  # Map instruction_following to coherence
            "truthfulness": "correctness"  # Map truthfulness to correctness
        }
        
        for ann_key, score_key in annotation_mapping.items():
            if ann_key in annotations:
                ann_data = annotations[ann_key]
                if isinstance(ann_data, dict) and "Rating" in ann_data:
                    try:
                        rating = float(ann_data["Rating"])
                        scores[score_key] = rating
                    except (ValueError, TypeError):
                        scores[score_key] = 0.0
                else:
                    scores[score_key] = 0.0
            else:
                scores[score_key] = 0.0
        
        return scores
    
    def process_single_row(example):
        """Process a single row with instruction and completions list."""
        instruction = example["instruction"]
        completions = example["completions"]
        
        # Validate instruction
        if not is_valid_text(instruction):
            return None
        
        # Language filtering for Chinese
        if not detect_language(instruction, ("zh",)):
            return None
        
        # Process each completion in this row
        processed_completions = []
        for completion in completions:
            if not isinstance(completion, dict):
                continue
            
            response = completion.get("response", "")
            if not is_valid_text(response):
                continue
            
            # Extract scores from annotations
            annotations = completion.get("annotations", {})
            scores = extract_scores_from_annotations(annotations)
            
            # Use overall_score if available, otherwise calculate from annotations
            overall_score = completion.get("overall_score", 0.0)
            if overall_score > 0:
                # Use overall_score as the primary score
                primary_score = overall_score
            else:
                # Calculate aggregated score from annotations
                primary_score, _ = aggregate_scores(scores)
            
            processed_completions.append({
                "instruction": instruction,
                "response": response,
                "score": primary_score,
                "scores": scores,
                "overall_score": overall_score,
                "model": completion.get("model", "unknown")
            })
        
        # Create preference pairs from this single row
        if len(processed_completions) < 2:
            return []
        
        # Sort by score (descending) - highest first
        processed_completions.sort(key=lambda x: x["score"], reverse=True)
        
        # Find best and worst responses
        best = processed_completions[0]  # Highest score
        worst = processed_completions[-1]  # Lowest score
        
        # Skip if responses are too similar
        if best["response"].strip() == worst["response"].strip():
            return []
        
        # Skip if length difference is too large
        if abs(len(best["response"]) - len(worst["response"])) > CONFIG["filters"]["max_length_delta"]:
            return []
        
        # Create preference pair
        return [{
            "prompt": best["instruction"],
            "chosen": best["response"],
            "rejected": worst["response"]
        }]
    
    # Process each row individually
    all_pairs = []
    for example in dataset:
        pairs = process_single_row(example)
        if pairs:
            all_pairs.extend(pairs)
    
    result = Dataset.from_list(all_pairs)
    print(f"{dataset_name} processed: {len(result)} pairs")
    return result


# ========== Dataset Operations ==========
def deduplicate_dataset(dataset: Dataset, jaccard_threshold: float = 0.9) -> Dataset:
    """Remove duplicate pairs using built-in dataset operations."""
    def jaccard_similarity(text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)
    
    def is_duplicate(example):
        """Check if example is a duplicate."""
        prompt = example["prompt"].strip()
        chosen = example["chosen"].strip()
        rejected = example["rejected"].strip()
        
        # Check if chosen and rejected are too similar
        if jaccard_similarity(chosen, rejected) >= jaccard_threshold:
            return False
        
        return True
    
    # Filter duplicates
    deduplicated = dataset.filter(is_duplicate)
    
    # Remove exact prompt duplicates (keep first occurrence)
    seen_prompts = set()
    def is_unique_prompt(example):
        prompt = example["prompt"].strip()
        if prompt in seen_prompts:
            return False
        seen_prompts.add(prompt)
        return True
    
    final_dataset = deduplicated.filter(is_unique_prompt)
    
    print(f"Deduplication: {len(dataset)} -> {len(final_dataset)} pairs")
    return final_dataset

def save_datasets(helpsteer2_data: Dataset, ultra_cn_data: Dataset, combined_data: DatasetDict):
    """Save datasets to separate directories and generate JSONL files."""
    # Create output directories
    for dir_name in CONFIG["output_dirs"].values():
        os.makedirs(dir_name, exist_ok=True)
    
    # Save individual datasets
    helpsteer2_data.save_to_disk(CONFIG["output_dirs"]["helpsteer2"])
    ultra_cn_data.save_to_disk(CONFIG["output_dirs"]["ultra_cn"])
    combined_data.save_to_disk(CONFIG["output_dirs"]["combined"])
    
    # Generate JSONL files for visualization
    print("\nGenerating JSONL files for visualization...")
    
    # HelpSteer2 JSONL
    helpsteer2_jsonl_path = f"{CONFIG['output_dirs']['helpsteer2']}/helpsteer2_pairs.jsonl"
    with open(helpsteer2_jsonl_path, 'w', encoding='utf-8') as f:
        for example in helpsteer2_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"  HelpSteer2 JSONL: {helpsteer2_jsonl_path}")
    
    # UltraFeedback-CN JSONL
    ultra_cn_jsonl_path = f"{CONFIG['output_dirs']['ultra_cn']}/ultra_cn_pairs.jsonl"
    with open(ultra_cn_jsonl_path, 'w', encoding='utf-8') as f:
        for example in ultra_cn_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"  UltraFeedback-CN JSONL: {ultra_cn_jsonl_path}")
    
    # Combined train JSONL
    train_jsonl_path = f"{CONFIG['output_dirs']['combined']}/train.jsonl"
    with open(train_jsonl_path, 'w', encoding='utf-8') as f:
        for example in combined_data["train"]:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"  Train JSONL: {train_jsonl_path}")
    
    # Combined eval JSONL
    eval_jsonl_path = f"{CONFIG['output_dirs']['combined']}/eval.jsonl"
    with open(eval_jsonl_path, 'w', encoding='utf-8') as f:
        for example in combined_data["eval"]:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"  Eval JSONL: {eval_jsonl_path}")
    
    print(f"\nDatasets saved to:")
    print(f"  HelpSteer2: {CONFIG['output_dirs']['helpsteer2']}")
    print(f"  UltraFeedback-CN: {CONFIG['output_dirs']['ultra_cn']}")
    print(f"  Combined: {CONFIG['output_dirs']['combined']}")

# ========== Main Processing Pipeline ==========
def main():
    """Main processing pipeline."""
    print("=" * 60)
    print("🚀 DPO Dataset Processing Pipeline")
    print("=" * 60)
    
    # Print sampling configuration
    print(f"\nSampling Configuration:")
    print(f"  Target total samples: {CONFIG['sampling']['total_samples']}")
    print(f"  English samples: {CONFIG['sampling']['en_samples']} ({CONFIG['sampling']['en_ratio']*100:.0f}%)")
    print(f"  Chinese samples: {CONFIG['sampling']['zh_samples']} ({CONFIG['sampling']['zh_ratio']*100:.0f}%)")
    
    # Process HelpSteer2 (English data)
    print(f"\nProcessing HelpSteer2 (English data)...")
    helpsteer2_data = process_helpsteer2(CONFIG["datasets"]["helpsteer2"])
    
    # Process UltraFeedback-Chinese (Chinese data)
    print(f"\nProcessing UltraFeedback-Chinese (Chinese data)...")
    try:
        ultra_cn_data = process_ultra_cn(CONFIG["datasets"]["ultra_cn"])
    except Exception as e:
        print(f"Failed to load UltraFeedback-Chinese: {e}")
        print("Trying fallback dataset...")
        ultra_cn_data = process_ultra_cn(CONFIG["datasets"]["ultra_cn_fallback"])
        print(f"Fallback dataset loaded: {len(ultra_cn_data)} pairs")
    
    # Combine datasets
    print("\nCombining datasets...")
    combined = concatenate_datasets([helpsteer2_data, ultra_cn_data])
    
    # Deduplicate
    print("\nDeduplicating...")
    combined = deduplicate_dataset(combined, CONFIG["filters"]["jaccard_threshold"])
    
    # Final filtering
    print("\nFinal filtering...")
    combined = combined.filter(lambda x: all(is_valid_text(x[field]) for field in ["prompt", "chosen", "rejected"]))
    
    # Split into train/eval
    print("\nSplitting train/eval...")
    split_data = combined.train_test_split(
        test_size=CONFIG["filters"]["test_size"], 
        seed=42
    )
    
    final_dataset = DatasetDict({
        "train": split_data["train"].shuffle(seed=42),
        "eval": split_data["test"]
    })
    
    # Save all datasets
    print("\nSaving datasets...")
    save_datasets(helpsteer2_data, ultra_cn_data, final_dataset)
    
    # Print statistics
    print("\n📈 Final Statistics:")
    print(f"  HelpSteer2 pairs: {len(helpsteer2_data)}")
    print(f"  UltraFeedback-CN pairs: {len(ultra_cn_data)}")
    print(f"  Combined train: {len(final_dataset['train'])}")
    print(f"  Combined eval: {len(final_dataset['eval'])}")
    print(f"  Total pairs: {len(helpsteer2_data) + len(ultra_cn_data)}")
    
    # Calculate actual ratios
    total_pairs = len(helpsteer2_data) + len(ultra_cn_data)
    if total_pairs > 0:
        en_ratio = len(helpsteer2_data) / total_pairs
        zh_ratio = len(ultra_cn_data) / total_pairs
        print(f"  Actual English ratio: {en_ratio:.1%}")
        print(f"  Actual Chinese ratio: {zh_ratio:.1%}")
    
    print("\n✅ Processing completed successfully!")

if __name__ == "__main__":
    main()
