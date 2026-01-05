#!/usr/bin/env python3
"""
Interactive Model Testing Script
===============================

Test your trained Qwen3-4B model with an interactive chat interface.
"""

import os
import sys
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from config import get_output_config


class ModelTester:
    """Class for testing the trained model."""
    
    def __init__(self, model_path: str = get_output_config()["final_model_dir"]):
        """Initialize the model tester."""
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the trained model and tokenizer."""
        print(f"🔄 Loading model from: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        try:
            # Load model and tokenizer
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                str(self.model_path),
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            
            # Enable inference mode
            FastLanguageModel.for_inference(self.model)
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response for a given prompt."""
        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def run_predefined_tests(self):
        """Run tests on predefined prompts."""
        print("\n" + "="*60)
        print("📝 Running predefined tests...")
        print("="*60)
        
        test_cases = [
            "小明有10个苹果，吃了3个，还剩几个？",
            "What is the capital of France?",
            "用Python写一个Hello World程序",
            "解释什么是机器学习？",
            "Mary has 5 cats. She gives 2 cats to her friend. How many cats does she have left?",
            "什么是深度学习？请用简单的话解释。",
        ]
        
        for i, prompt in enumerate(test_cases, 1):
            print(f"\n--- Test {i} ---")
            print(f"🤔 Input: {prompt}")
            
            try:
                response = self.generate_response(prompt, max_new_tokens=256)
                print(f"🤖 Output: {response}")
                
            except Exception as e:
                print(f"❌ Error generating response: {e}")
        
        print("\n" + "="*60)
    
    def interactive_chat(self):
        """Start interactive chat session."""
        print("\n" + "="*60)
        print("💬 Interactive Chat Mode")
        print("Commands:")
        print("  - Type your message and press Enter")
        print("  - Type 'quit', 'exit', or 'bye' to exit")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'help' to show this help")
        print("="*60)
        
        conversation = []
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                    
                if user_input.lower() == 'clear':
                    conversation = []
                    print("🗑️  Conversation history cleared!")
                    continue
                    
                if user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  quit/exit/bye - Exit chat")
                    print("  clear - Clear conversation history")
                    print("  help - Show this help")
                    continue
                
                # Add to conversation
                conversation.append({"role": "user", "content": user_input})
                
                # Generate response
                print("🤖 Assistant: ", end="", flush=True)
                
                # Use full conversation for context
                text = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][len(inputs.input_ids[0]):], 
                    skip_special_tokens=True
                ).strip()
                
                print(response)
                
                # Add assistant response to conversation
                conversation.append({"role": "assistant", "content": response})
                
                # Keep conversation manageable
                if len(conversation) > 20:
                    conversation = conversation[-20:]
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

def main():
    """Main function."""
    print("🤖 Qwen3-4B Model Tester")
    print("="*40)
    
    # Initialize tester
    tester = ModelTester()
    
    try:
        # Load model
        tester.load_model()
        
        # Show menu
        while True:
            print("\n🔧 What would you like to do?")
            print("1. Run predefined tests")
            print("2. Interactive chat")
            print("3. Both")
            print("4. Exit")
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                tester.run_predefined_tests()
                
            elif choice == '2':
                tester.interactive_chat()
                
            elif choice == '3':
                tester.run_predefined_tests()
                tester.interactive_chat()
                
            elif choice == '4':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-4.")
                
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please train the model first by running: python train.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
