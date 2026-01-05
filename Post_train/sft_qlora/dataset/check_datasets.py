#!/usr/bin/env python3
"""
Check the format of the Arrow dataset
"""
import os
import json
from datasets import load_dataset, load_from_disk
from pathlib import Path

def check_raw_data(file_path: str, file_type: str = "json", limit: int = 3):
    """Check the format of the raw data file"""
    print(f"\n=== Check raw data: {file_path} ===")
    
    try:
        if file_type == "json":
            ds = load_dataset("json", data_files=file_path, split="train")
        elif file_type == "csv":
            ds = load_dataset("csv", data_files=file_path, split="train")
        else:
            print(f"Unsupported file type: {file_type}")
            return
        
        print(f"Dataset size: {len(ds)} records")
        print(f"Column names: {ds.column_names}")
        print(f"Dataset info: {ds.info}")
        
        # Show the first few samples
        for i in range(min(limit, len(ds))):
            print(f"\n--- Sample {i+1} ---")
            sample = ds[i]
            for key, value in sample.items():
                print(f"{key}: {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}")
                
    except Exception as e:
        print(f"Error reading data: {e}")

def check_arrow_data(arrow_path: str, limit: int = 3):
    """Check the format of the processed Arrow dataset"""
    print(f"\n=== Check Arrow data: {arrow_path} ===")
    
    try:
        ds = load_from_disk(arrow_path)
        print(f"Dataset size: {len(ds)} records")
        print(f"Column names: {ds.column_names}")
        
        # Check the data structure
        if len(ds) > 0:
            first_sample = ds[0]
            print(f"\nThe structure of the first sample:")
            for key, value in first_sample.items():
                print(f"  {key}: {type(value)} - {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
        
            # Show the first few samples
        for i in range(min(limit, len(ds))):
            print(f"\n--- Sample {i+1} ---")
            sample = ds[i]
            print(json.dumps(sample, ensure_ascii=False, indent=2)[:500])
            if len(json.dumps(sample, ensure_ascii=False, indent=2)) > 500:
                print("...")
                
    except Exception as e:
        print(f"Error reading Arrow data: {e}")

def check_data_integrity(ds, expected_fields=None):
    """Check the data integrity"""
    print(f"\n=== Data integrity check ===")
    
    if expected_fields is None:
        expected_fields = ["messages", "source", "tags"]
    
    missing_fields = []
    for field in expected_fields:
        if field not in ds.column_names:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Missing fields: {missing_fields}")
    else:
        print(f"✅ All required fields exist: {expected_fields}")
    
    # Check the structure of the messages field
    if "messages" in ds.column_names and len(ds) > 0:
        print(f"\nCheck the structure of the messages field...")
        for i in range(min(3, len(ds))):
            messages = ds[i]["messages"]
            if not isinstance(messages, list):
                print(f"❌ Sample{i+1}: messages is not a list")
                continue
                
            valid_message = True
            for msg in messages:
                if not isinstance(msg, dict):
                    print(f"❌ Sample{i+1}: message is not a dictionary")
                    valid_message = False
                    break
                if "role" not in msg or "content" not in msg:
                    print(f"❌ Sample{i+1}: message missing role or content field")
                    valid_message = False
                    break
            
            if valid_message:
                print(f"✅ Sample{i+1}: messages format is correct")

def main():
    """Main function"""
    base_dir = Path("base_datasets")
    processed_dir = Path("processed_datasets/normalized")
    
    print("🔍 Dataset format check tool")
    print("=" * 50)
    
    # 1. Check the raw data
    datasets = [
        ("cmath", base_dir / "cmath" / "cmath_dev.jsonl", "json"),
        ("emoji", base_dir / "DPO-zh-en-emoji" / "merged_dpo_zh_emoji_for_firefly.jsonl", "json"),
        ("mawps", base_dir / "MAWPS" / "MAWPS.csv", "csv"),
    ]
    
    for name, path, ftype in datasets:
        if path.exists():
            check_raw_data(str(path), ftype)
        else:
            print(f"\n❌ Raw data file does not exist: {path}")
    
    # 2. Check the processed data
    print("\n" + "="*50)
    print("Check the processed Arrow data")
    
    for name, _, _ in datasets:
        arrow_path = processed_dir / name
        if arrow_path.exists():
            check_arrow_data(str(arrow_path))
            
            # Load the dataset for integrity check
            try:
                ds = load_from_disk(str(arrow_path))
                check_data_integrity(ds)
            except Exception as e:
                print(f"❌ Error loading {name} dataset: {e}")
        else:
            print(f"\n❌ Processed data does not exist: {arrow_path}")

if __name__ == "__main__":
    main()
