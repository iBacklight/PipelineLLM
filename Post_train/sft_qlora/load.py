from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import os
import sys

model_name = "unsloth/Qwen3-4B-Instruct-2507"
models_dir = "../../../models"
save_dir = os.path.join(models_dir, "unsloth/Qwen3/Qwen3-4B-Instruct-2507/")
save_dir = os.path.abspath(save_dir)
print(save_dir)


TEST = True
VLLM_TEST = False

# load the tokenizer and the model (only for local testing)
if TEST:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        cache_dir=save_dir
    )

for name, param in model.named_parameters():
    print(name, param.dtype)
    
# For a simple output
if TEST:
    # prepare the model input
    prompt = "你好，能介绍一下你自己吗？"
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True, # there is no need for setting enable_thinking
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=16384
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("content:", content)

if VLLM_TEST:
    print("=== Testing vLLM API Server ===")
    
    # Initialize OpenAI client to point to local vLLM server
    client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't require a real API key
        base_url="http://localhost:8000/v1",  # vLLM server endpoint
    )
    
    print("Getting available models from vLLM server...")
    try:
        models = client.models.list()
        if models.data:
            vllm_model_name = models.data[0].id 
            print(f"Using model: {vllm_model_name}")
        else:
            print("No models available on vLLM server!")
            exit(1)
    except Exception as e:
        print(f"Error getting model list: {e}")
        exit(1)
    
    # Test 1: Simple chat completion
    print("\n1. Testing basic chat completion...")
    try:
        response = client.chat.completions.create(
            model=vllm_model_name, 
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Give me a short introduction to large language models."}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        
        print(f"Response: {response.choices[0].message.content}")
        print(f"Usage: {response.usage}")
        
    except Exception as e:
        print(f"Error in basic chat completion: {e}")
    
    # Test 2: Streaming response
    print("\n2. Testing streaming response...")
    try:
        stream = client.chat.completions.create(
            model=vllm_model_name,
            messages=[
                {"role": "user", "content": "Explain what is machine learning in 3 sentences."}
            ],
            max_tokens=300,
            temperature=0.8,
            stream=True,
        )
        
        print("Streaming response:")
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")
        
    except Exception as e:
        print(f"Error in streaming: {e}")
    
    # Test 3: Multiple conversation turns
    print("\n3. Testing multi-turn conversation...")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
        ]
        
        # First response
        response1 = client.chat.completions.create(
            model=vllm_model_name,
            messages=messages,
            max_tokens=200,
            temperature=0.5,
        )
        
        print(f"Assistant: {response1.choices[0].message.content}")
        
        # Add assistant's response and user's follow-up
        messages.append({"role": "assistant", "content": response1.choices[0].message.content})
        messages.append({"role": "user", "content": "Can you give me a simple Python code example?"})
        
        # Second response
        response2 = client.chat.completions.create(
            model=vllm_model_name,
            messages=messages,
            max_tokens=300,
            temperature=0.5,
        )
        
        print(f"Assistant: {response2.choices[0].message.content}")
        
    except Exception as e:
        print(f"Error in multi-turn conversation: {e}")
    
    # Test 4: Different sampling parameters
    print("\n4. Testing different sampling parameters...")
    try:
        test_prompt = "Write a creative story opening in one sentence."
        
        # High temperature (more creative)
        response_creative = client.chat.completions.create(
            model=vllm_model_name,
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=100,
            temperature=1.2,
            top_p=0.9,
        )
        
        # Low temperature (more deterministic)
        response_focused = client.chat.completions.create(
            model=vllm_model_name,
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=100,
            temperature=0.1,
            top_p=0.1,
        )
        
        print(f"Creative (temp=1.2): {response_creative.choices[0].message.content}")
        print(f"Focused (temp=0.1): {response_focused.choices[0].message.content}")
        
    except Exception as e:
        print(f"Error in parameter testing: {e}")
        
    print("\n=== vLLM API Testing Complete ===")
