#!/usr/bin/env python3
"""
Evaluation script for trained Reward Model.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from reward_model_utils import (
    RewardModel,
    RewardModelEvaluator,
    load_preference_dataset,
    load_reward_model
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardModelEvaluation:
    """Comprehensive evaluation for Reward Model."""
    
    def __init__(self, model_path: str, model_name: str, device: str = "cuda"):
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        
        # Load model and tokenizer
        self.model, self.tokenizer = load_reward_model(model_name, model_path, device)
        
        # Initialize evaluator
        self.evaluator = RewardModelEvaluator(
            self.model, 
            self.tokenizer, 
            self.device
        )
    
    def evaluate_preference_accuracy(
        self, 
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate preference prediction accuracy."""
        logger.info("Evaluating preference accuracy...")
        return self.evaluator.evaluate_preference_accuracy(test_data)
    
    def evaluate_reward_distribution(
        self, 
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate reward distribution statistics."""
        logger.info("Evaluating reward distribution...")
        return self.evaluator.evaluate_reward_distribution(test_data)
    
    def evaluate_on_multiple_datasets(self, dataset_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Evaluate on multiple datasets."""
        results = {}
        
        for dataset_name in dataset_names:
            logger.info(f"Evaluating on dataset: {dataset_name}")
            
            try:
                # Load test data
                test_data = load_preference_dataset(dataset_name, "test")
                
                # Limit data size for faster evaluation
                if len(test_data) > 1000:
                    test_data = test_data[:1000]
                
                # Evaluate
                accuracy_metrics = self.evaluate_preference_accuracy(test_data)
                distribution_metrics = self.evaluate_reward_distribution(test_data)
                
                results[dataset_name] = {
                    **accuracy_metrics,
                    **distribution_metrics
                }
                
                logger.info(f"Results for {dataset_name}: {results[dataset_name]}")
                
            except Exception as e:
                logger.error(f"Error evaluating on {dataset_name}: {e}")
                results[dataset_name] = {"error": str(e)}
        
        return results
    
    def analyze_reward_patterns(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze reward patterns and correlations."""
        logger.info("Analyzing reward patterns...")
        
        chosen_rewards = []
        rejected_rewards = []
        response_lengths_chosen = []
        response_lengths_rejected = []
        
        with torch.no_grad():
            for item in tqdm(test_data, desc="Analyzing patterns"):
                prompt = item.get("prompt", "")
                chosen = item.get("chosen", "")
                rejected = item.get("rejected", "")
                
                if not chosen or not rejected:
                    continue
                
                # Get rewards
                chosen_text = prompt + chosen
                rejected_text = prompt + rejected
                
                chosen_tokens = self.tokenizer(
                    chosen_text,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)
                
                rejected_tokens = self.tokenizer(
                    rejected_text,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)
                
                chosen_output = self.model(
                    chosen_tokens["input_ids"],
                    chosen_tokens["attention_mask"]
                )
                rejected_output = self.model(
                    rejected_tokens["input_ids"],
                    rejected_tokens["attention_mask"]
                )
                
                chosen_reward = chosen_output["rewards"].item()
                rejected_reward = rejected_output["rewards"].item()
                
                chosen_rewards.append(chosen_reward)
                rejected_rewards.append(rejected_reward)
                
                # Response lengths
                response_lengths_chosen.append(len(chosen.split()))
                response_lengths_rejected.append(len(rejected.split()))
        
        # Convert to numpy arrays
        chosen_rewards = np.array(chosen_rewards)
        rejected_rewards = np.array(rejected_rewards)
        response_lengths_chosen = np.array(response_lengths_chosen)
        response_lengths_rejected = np.array(response_lengths_rejected)
        
        # Compute correlations
        reward_length_corr_chosen = np.corrcoef(chosen_rewards, response_lengths_chosen)[0, 1]
        reward_length_corr_rejected = np.corrcoef(rejected_rewards, response_lengths_rejected)[0, 1]
        
        # Compute reward difference
        reward_differences = chosen_rewards - rejected_rewards
        
        return {
            "chosen_rewards_mean": np.mean(chosen_rewards),
            "chosen_rewards_std": np.std(chosen_rewards),
            "rejected_rewards_mean": np.mean(rejected_rewards),
            "rejected_rewards_std": np.std(rejected_rewards),
            "reward_difference_mean": np.mean(reward_differences),
            "reward_difference_std": np.std(reward_differences),
            "reward_length_corr_chosen": reward_length_corr_chosen,
            "reward_length_corr_rejected": reward_length_corr_rejected,
            "preference_accuracy": np.mean(chosen_rewards > rejected_rewards),
            "num_samples": len(chosen_rewards)
        }
    
    def generate_evaluation_report(
        self, 
        test_data: List[Dict[str, Any]], 
        output_file: str = "evaluation_report.json"
    ):
        """Generate comprehensive evaluation report."""
        logger.info("Generating evaluation report...")
        
        # Run all evaluations
        accuracy_metrics = self.evaluate_preference_accuracy(test_data)
        distribution_metrics = self.evaluate_reward_distribution(test_data)
        pattern_metrics = self.analyze_reward_patterns(test_data)
        
        # Combine all metrics
        report = {
            "accuracy_metrics": accuracy_metrics,
            "distribution_metrics": distribution_metrics,
            "pattern_metrics": pattern_metrics,
            "model_info": {
                "model_name": self.model_name,
                "model_path": self.model_path,
                "device": str(self.device),
                "num_test_samples": len(test_data)
            }
        }
        
        # Save report
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Preference Accuracy: {accuracy_metrics['accuracy']:.4f}")
        print(f"Mean Chosen Reward: {distribution_metrics['mean_reward']:.4f}")
        print(f"Reward Difference: {pattern_metrics['reward_difference_mean']:.4f}")
        print(f"Preference Accuracy (Pattern): {pattern_metrics['preference_accuracy']:.4f}")
        print("="*50)
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Reward Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Base model name")
    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf", help="Dataset to evaluate on")
    parser.add_argument("--output_file", type=str, default="evaluation_report.json", help="Output file for report")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum samples to evaluate")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = RewardModelEvaluation(
        args.model_path, 
        args.model_name, 
        args.device
    )
    
    # Load test data
    test_data = load_preference_dataset(args.dataset_name, "test")
    
    # Limit samples if specified
    if args.max_samples > 0 and len(test_data) > args.max_samples:
        test_data = test_data[:args.max_samples]
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(test_data, args.output_file)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
