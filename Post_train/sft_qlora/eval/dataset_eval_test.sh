#!/bin/bash

# EvalScope Dataset Evaluation Test
# Testing with open-source datasets like MMLU Pro

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== EvalScope Dataset Evaluation Test ===${NC}"

# Activate conda environment
echo -e "${BLUE}Activating llm conda environment...${NC}"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llm

# Check if evalscope is installed
if ! command -v evalscope &> /dev/null; then
    echo -e "${RED}Error: evalscope not installed${NC}"
    echo "Please run: conda activate llm && python -m pip install evalscope[perf] -U"
    exit 1
fi

# Check if vLLM server is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${RED}Error: vLLM server not running on localhost:8000${NC}"
    echo "Please ensure vLLM server is running"
    exit 1
fi

echo -e "${GREEN}✓ All checks passed${NC}"

# Model configuration for evaluation
MODEL_NAME="/home/awpc/studies/models/unsloth/Qwen3/Qwen3-4B-Instruct-2507/models--unsloth--Qwen3-4B-Instruct-2507/snapshots/992063681dc2f7de4ee976110199552935cad284"

echo -e "${BLUE}Using model: $(basename $MODEL_NAME)${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Create results directory
RESULTS_DIR="$SCRIPT_DIR/evalscope_dataset_results/base_model"
mkdir -p "$RESULTS_DIR"

echo -e "\n${YELLOW}Available dataset options for 4B models:${NC}"
echo "1. MMLU Pro - Multi-task Language Understanding (Professional)"
echo "2. HellaSwag - Common sense reasoning"
echo "3. ARC - AI2 Reasoning Challenge"
echo "4. GSM8K - Grade School Math"
echo ""

# Function to run dataset evaluation
run_dataset_eval() {
    local dataset_name=$1
    local dataset_args=${2:-"{}"}
    
    echo -e "\n${BLUE}=== Testing Dataset: $dataset_name ===${NC}"
    
    echo "Running evalscope eval for $dataset_name..."
    
    evalscope eval \
      --model "$MODEL_NAME" \
      --datasets "$dataset_name" \
      --dataset-args "$dataset_args" \
      --limit 3 \
      --eval-batch-size 1 \
      --work-dir "$RESULTS_DIR/${dataset_name}_results" \
      --api-url "http://localhost:8000/v1" \
      --api-key "EMPTY" \
      --debug
}

# Test 1: MMLU Pro (subset to keep it manageable for 4B model)
# echo -e "\n${BLUE}=== Test 1: MMLU Pro (subset) ===${NC}"
# run_dataset_eval "mmlu_pro" '{"subset": "high_school_mathematics"}'

# Test 2: HellaSwag (common sense reasoning) - using minimal memory
echo -e "\n${BLUE}=== Test 2: HellaSwag (Low Memory) ===${NC}"
# run_dataset_eval "hellaswag" '{}'

# echo -e "\n${BLUE}=== Alternative: Simple ARC-Easy Test ===${NC}"
# echo "If HellaSwag still uses too much memory, trying ARC-Easy..."
# Uncomment the line below if HellaSwag fails
run_dataset_eval "arc" '{"subset": "ARC-Easy"}'

# # Test 3: ARC Easy (reasoning)
# echo -e "\n${BLUE}=== Test 3: ARC Easy ===${NC}"
# run_dataset_eval "arc" '{\"subset\": \"ARC-Easy\"}'

# # Test 4: GSM8K (math reasoning) - smaller subset
# echo -e "\n${BLUE}=== Test 4: GSM8K (sample) ===${NC}"
# run_dataset_eval "gsm8k" '{}'

# echo -e "\n${GREEN}=== All Dataset Evaluations Completed ===${NC}"
# echo -e "${BLUE}Results saved in: $RESULTS_DIR${NC}"

# # Show results summary
# echo -e "\n${BLUE}=== Results Summary ===${NC}"
# find "$RESULTS_DIR" -name "*.json" -o -name "*.txt" | head -10
# echo ""
# echo -e "${YELLOW}To view detailed results, check the files in: $RESULTS_DIR${NC}"

# # Create a simple results aggregator
# echo -e "\n${BLUE}=== Creating Results Summary ===${NC}"
# cat > "$RESULTS_DIR/evaluation_summary.md" << EOF
# # Dataset Evaluation Results Summary

# **Model**: Qwen3-4B-Instruct-2507
# **Test Date**: $(date)
# **Test Configuration**: 
# - API: OpenAI compatible (vLLM)
# - Endpoint: http://localhost:8000/v1
# - Sample limit: 50 questions per dataset

# ## Tested Datasets

# 1. **MMLU Pro** (High School Mathematics)
#    - Focus: Professional-level multi-task understanding
#    - Subset: Mathematics questions
   
# 2. **HellaSwag** 
#    - Focus: Common sense reasoning
#    - Task: Sentence completion with context
   
# 3. **ARC Easy**
#    - Focus: Science reasoning
#    - Task: Multiple choice science questions
   
# 4. **GSM8K**
#    - Focus: Mathematical reasoning
#    - Task: Grade school math word problems

# ## How to View Results

# Check the individual result directories:
# - \`mmlu_pro_results/\` - MMLU Pro evaluation
# - \`hellaswag_results/\` - HellaSwag evaluation  
# - \`arc_results/\` - ARC evaluation
# - \`gsm8k_results/\` - GSM8K evaluation

# Each directory contains:
# - Detailed JSON results
# - Performance metrics
# - Sample predictions

# ## Next Steps

# 1. Analyze accuracy scores for each dataset
# 2. Compare with baseline model performance
# 3. Identify areas for improvement
# 4. Consider fine-tuning on specific domains

# EOF

# echo -e "${GREEN}✓ Summary report created: $RESULTS_DIR/evaluation_summary.md${NC}"
