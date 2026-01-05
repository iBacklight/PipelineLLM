#!/usr/bin/env python3
"""
Train a Reward Model using trl library. This script is from https://huggingface.co/docs/trl/main/en/reward_trainer
Based on Qwen 0.6B model.
"""
from trl import RewardTrainer
from datasets import load_dataset

trainer = RewardTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
)
trainer.train()