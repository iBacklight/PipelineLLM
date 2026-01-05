#!/bin/bash

# EvalScope Performance Testing Script
# Testing local vLLM deployed Qwen3-4B-Instruct-2507 model

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color
CONDA_NAME="llm"

echo -e "${BLUE}=== EvalScope Performance Testing Started ===${NC}"

# Activate conda environment
echo -e "${BLUE}Activating $CONDA_NAME conda environment...${NC}"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_NAME

# Check if evalscope is installed
if ! command -v evalscope &> /dev/null; then
    echo -e "${RED}Error: evalscope not installed${NC}"
    echo "Please run: conda activate $CONDA_NAME && python -m pip install evalscope[perf] -U"
    exit 1
fi

# Check if vLLM server is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${RED}Error: vLLM server not running on localhost:8000${NC}"
    echo "Please ensure vLLM server is running"
    exit 1
fi

echo -e "${GREEN}vLLM server detected and running${NC}"

# Get model related paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../../models"
MODEL_NAME="/home/awpc/studies/models/unsloth/Qwen3/Qwen3-4B-Instruct-2507/models--unsloth--Qwen3-4B-Instruct-2507/snapshots/992063681dc2f7de4ee976110199552935cad284"
# Use the same path for tokenizer since they're in the same directory
TOKENIZER_PATH="$MODEL_NAME"

echo -e "${BLUE}Using model path: $MODEL_NAME${NC}"
echo -e "${BLUE}Using tokenizer path: $TOKENIZER_PATH${NC}"

# Create results directory
RESULTS_DIR="$SCRIPT_DIR/evalscope_perf_results/base_model"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}=== Test 1: Light Load Performance Test ===${NC}"
evalscope perf \
  --url 'http://127.0.0.1:8000/v1/chat/completions' \
  --parallel 5\
  --number 10 \
  --model "$MODEL_NAME" \
  --api openai \
  --dataset openqa \
  --max-tokens 512 \
  --min-tokens 256 \
  --prefix-length 0 \
  --min-prompt-length 50 \
  --max-prompt-length 512 \
  --tokenizer-path "$TOKENIZER_PATH" \
  --extra-args '{"ignore_eos": true}' \
  --output "$RESULTS_DIR/light_load_test.json"

# echo -e "\n${BLUE}=== Test 2: Medium Load Performance Test ===${NC}"
# evalscope perf \
#   --url 'http://127.0.0.1:8000/v1/chat/completions' \
#   --parallel 10 20 50 \
#   --number 50 100 200 \
#   --model "$MODEL_NAME" \
#   --api openai \
#   --dataset random \
#   --max-tokens 1024 \
#   --min-tokens 512 \
#   --prefix-length 0 \
#   --min-prompt-length 100 \
#   --max-prompt-length 1024 \
#   --tokenizer-path "$TOKENIZER_PATH" \
#   --extra-args '{"ignore_eos": true}' \
#   --output "$RESULTS_DIR/medium_load_test.json"

# echo -e "\n${BLUE}=== Test 3: High Load Performance Test ===${NC}"
# evalscope perf \
#   --url 'http://127.0.0.1:8000/v1/chat/completions' \
#   --parallel 50 100 200 \
#   --number 100 200 500 \
#   --model "$MODEL_NAME" \
#   --api openai \
#   --dataset random \
#   --max-tokens 2048 \
#   --min-tokens 1024 \
#   --prefix-length 0 \
#   --min-prompt-length 200 \
#   --max-prompt-length 2048 \
#   --tokenizer-path "$TOKENIZER_PATH" \
#   --extra-args '{"ignore_eos": true}' \
#   --output "$RESULTS_DIR/high_load_test.json"

# echo -e "\n${BLUE}=== Test 4: Streaming Response Performance Test ===${NC}"
# evalscope perf \
#   --url 'http://127.0.0.1:8000/v1/chat/completions' \
#   --parallel 1 5 10 20 \
#   --number 20 50 100 \
#   --model "$MODEL_NAME" \
#   --api openai \
#   --dataset random \
#   --max-tokens 1024 \
#   --min-tokens 512 \
#   --prefix-length 0 \
#   --min-prompt-length 100 \
#   --max-prompt-length 1024 \
#   --tokenizer-path "$TOKENIZER_PATH" \
#   --extra-args '{"ignore_eos": true, "stream": true}' \
#   --output "$RESULTS_DIR/streaming_test.json"

# echo -e "\n${GREEN}=== All Performance Tests Completed ===${NC}"
# echo -e "${BLUE}Test results saved in: $RESULTS_DIR${NC}"
# echo -e "${BLUE}File list:${NC}"
# ls -la "$RESULTS_DIR"

# echo -e "\n${BLUE}=== Generating Performance Report ===${NC}"
# cat > "$RESULTS_DIR/test_summary.md" << EOF
# # Qwen3-4B-Instruct-2507 Performance Test Report

# Test Time: $(date)
# Model: Qwen3-4B-Instruct-2507
# Server: vLLM (localhost:8000)

# ## Test Configuration

# - **Light Load Test**: Concurrency 1-10, Requests 10-50, Output 256-512 tokens
# - **Medium Load Test**: Concurrency 10-50, Requests 50-200, Output 512-1024 tokens  
# - **High Load Test**: Concurrency 50-200, Requests 100-500, Output 1024-2048 tokens
# - **Streaming Response Test**: Concurrency 1-20, Requests 20-100, Streaming Output 512-1024 tokens

# ## Test Result Files

# - light_load_test.json: Light load test results
# - medium_load_test.json: Medium load test results
# - high_load_test.json: High load test results
# - streaming_test.json: Streaming response test results

# ## Key Metrics Description

# - **Throughput**: Requests processed per second (req/s)
# - **Mean Latency**: Average response time (ms)
# - **P95/P99 Latency**: 95%/99% of requests response time below this value (ms)
# - **Error Rate**: Percentage of failed requests (%)
# - **Tokens/s**: Tokens generated per second

# EOF

# echo -e "${GREEN}Test report generated: $RESULTS_DIR/test_summary.md${NC}"
