# Qwen3-4B Fine-tuning with LoRA/QLoRA

A comprehensive, production-ready training pipeline for fine-tuning Qwen3-4B-Instruct using LoRA/QLoRA techniques.

## 🏗️ Project Structure

```
sft_qlora/
├── train.py           # Main training script (OOP design)
├── config.py          # Centralized configuration
├── run_training.py    # Simple training runner
├── test_model.py      # Interactive model testing
├── dataset/           # Dataset processing tools
│   ├── build_datasets.py
│   └── check_datasets.py
├── results/           # Training outputs
├── logs/              # Training logs
└── final_model/       # Saved model
```

## 🚀 Quick Start

### 1. Prepare Your Datasets

First, process your raw datasets:

```bash
cd dataset
python build_datasets.py
```

This will create normalized, deduplicated datasets in the proper format.

### 2. Configure Training

Edit `config.py` to adjust training parameters for your hardware:

```python
# For RTX 3090/4090 (24GB VRAM)
TRAINING_CONFIG.update({
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
})

# For RTX 3080/4080 (16GB VRAM)
TRAINING_CONFIG.update({
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
})
```

### 3. Start Training

**Option 1: Simple runner (recommended for beginners)**
```bash
python run_training.py
```

**Option 2: Direct training script**
```bash
python train.py
```

**Option 3: Advanced usage with custom config**
```python
from train import Qwen3Trainer

# Custom configuration
config_override = {
    "training": {"num_train_epochs": 3, "learning_rate": 1e-4},
    "lora": {"r": 32, "lora_alpha": 64}
}

trainer = Qwen3Trainer(config_override=config_override)
trainer.run_full_training_pipeline()
```

### 4. Test Your Model

After training, test your model:

```bash
python test_model.py
```

This provides:
- Predefined test cases
- Interactive chat interface
- Conversation with context

## 🔧 Configuration

The `config.py` file contains all training parameters organized by category:

- **MODEL_CONFIG**: Model loading settings
- **LORA_CONFIG**: LoRA hyperparameters
- **TRAINING_CONFIG**: Training hyperparameters
- **SFT_CONFIG**: SFT-specific parameters (packing, NEFTune, etc.)
- **TRAINER_CONFIG**: Trainer selection (SFTConfig vs TrainingArguments)
- **OUTPUT_CONFIG**: Output directories
- **DATA_CONFIG**: Dataset paths and settings

### 🎯 SFTConfig vs TrainingArguments

**SFTConfig**
- Built for supervised fine-tuning tasks
- Automatic packing for efficiency
- NEFTune noise injection support
- Optimized memory management
- Integrated formatting functions

**TrainingArguments**
- Full Transformers compatibility
- More granular control options
- Extensive logging capabilities
- Custom callback support

## 📊 Training Features

### Object-Oriented Design
- `Qwen3Trainer` class encapsulates all functionality
- Clean separation of concerns
- Easy to extend and modify
- Professional error handling and logging

### Key Features
- ✅ **Memory Efficient**: 4-bit quantization with LoRA
- ✅ **Flexible Configuration**: Easy hardware adaptation
- ✅ **Comprehensive Logging**: Detailed training logs
- ✅ **Automatic Evaluation**: Built-in model testing
- ✅ **Resume Training**: Checkpoint support
- ✅ **Error Handling**: Robust error recovery

### Training Pipeline
1. **Environment Setup**: Create directories, check CUDA
2. **Model Loading**: Load Qwen3-4B with quantization
3. **LoRA Setup**: Apply parameter-efficient fine-tuning
4. **Data Processing**: Format datasets with chat templates
5. **Training**: Execute supervised fine-tuning
6. **Evaluation**: Test model with sample prompts
7. **Saving**: Save model and tokenizer

## 🎯 Usage Examples

### Basic Training 
```python
from train import Qwen3Trainer

trainer = Qwen3Trainer()
trainer.run_full_training_pipeline()
```

### Using SFTConfig with Custom Settings
```python
config_override = {
    "trainer": {"use_sft_config": True},
    "sft": {
        "packing": True,
        "max_seq_length": 1024,
        "neftune_noise_alpha": 5,
        "dataset_num_proc": 8,
    },
    "training": {
        "num_train_epochs": 3,
        "learning_rate": 1e-4,
    }
}

trainer = Qwen3Trainer(config_override=config_override)
trainer.run_full_training_pipeline()
```

### Using TrainingArguments (Legacy Mode)
```python
config_override = {
    "trainer": {"use_sft_config": False},
    "training": {
        "num_train_epochs": 2,
        "learning_rate": 2e-4,
        "warmup_steps": 100,
    },
    "data": {"packing": False}
}

trainer = Qwen3Trainer(config_override=config_override)
trainer.run_full_training_pipeline()
```

### Custom Model Path
```python
trainer = Qwen3Trainer(model_path="/path/to/custom/model")
trainer.run_full_training_pipeline()
```

### Step-by-Step Training
```python
trainer = Qwen3Trainer()
trainer.setup_environment()
trainer.load_model_and_tokenizer()
trainer.setup_lora()
trainer.load_datasets()
trainer.setup_trainer()
trainer.train()
trainer.save_model()
```

### Checkpoint Management

**Minimal Storage Mode (Save Disk Space)**
```python
# Disable checkpoints to save storage
config_override = {
    "output": {"save_checkpoints": False}
}
trainer = Qwen3Trainer(config_override=config_override)
trainer.run_full_training_pipeline()
```

**Full Checkpoint Mode**
```python
# Enable full checkpoint saving
config_override = {
    "output": {"save_checkpoints": True},
    "training": {
        "save_steps": 100,
        "save_total_limit": 2,
        "eval_steps": 50,
    }
}
trainer = Qwen3Trainer(config_override=config_override)
trainer.run_full_training_pipeline()
```

**Checkpoint Examples**
```bash
# Explore all checkpoint options
python checkpoint_examples.py
```

## 📈 Monitoring Training

- **Real-time logs**: Check `training.log`
- **TensorBoard**: Enable in config for visual monitoring
- **Evaluation metrics**: Automatic evaluation every 100 steps
- **Memory usage**: CUDA memory tracking

## 🛠️ Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`

### Slow Training
- Enable gradient checkpointing
- Use bf16 instead of fp16
- Optimize dataloader settings

### Model Quality Issues
- Adjust learning rate
- Increase training epochs
- Modify LoRA parameters (r, alpha)

### Storage Issues
- **Limited disk space**: Enable minimal storage mode (`save_checkpoints: False`)
- **Want checkpoint resumption**: Use full checkpoint mode with `save_total_limit: 2`
- **Separate storage**: Set custom `checkpoint_dir` for checkpoints

## 📝 Advanced Configuration

### Custom LoRA Settings
```python
LORA_CONFIG = {
    "r": 32,                    # Higher = more parameters
    "lora_alpha": 64,           # Scaling factor
    "lora_dropout": 0.05,       # Regularization
    "target_modules": [         # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj"
    ]
}
```

### Hardware Optimization
```python
# For A100 (80GB)
GPU_A100_CONFIG = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "max_seq_length": 4096,
}
```

## 🎉 Results

After training, you'll have:
- Fine-tuned Qwen3-4B model in `./final_model/`
- Training logs with detailed metrics
- Evaluation results on test prompts
- Ready-to-use model for inference

## 📚 Next Steps

1. **Evaluate thoroughly** with your specific use cases
2. **Deploy** using vLLM or similar inference engines  
3. **Iterate** on hyperparameters for better performance
4. **Scale up** with larger datasets or longer training

Happy training! 🚀
