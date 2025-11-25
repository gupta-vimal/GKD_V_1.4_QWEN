# GKD v1.3 - Generalized Knowledge Distillation (Without PEFT)

Knowledge distillation training script for fine-tuning language models using Generalized Knowledge Distillation (GKD) with full fine-tuning (no PEFT/QLoRA).

## Features

- **Full Fine-Tuning**: Complete model training without parameter-efficient methods
- **Knowledge Distillation**: Teacher-student architecture (3B → 1B)
- **OpenOrca Dataset**: High-quality instruction-following dataset (20K prompts)
- **CUDA Support**: GPU-accelerated training with automatic device detection
- **No Quantization**: Full precision training for maximum model quality

## Models

- **Teacher Model**: `meta-llama/Llama-3.2-3B-Instruct`
- **Student Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Dataset**: `Open-Orca/OpenOrca` (20,000 prompts)

## Requirements

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- PyTorch with CUDA support

## Setup

### 1. Create Virtual Environment

**Windows:**
```powershell
setup_venv.bat
```

**Linux/Mac:**
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

### 2. Manual Setup

```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install PyTorch with CUDA (if needed)
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Verify CUDA

```powershell
python check_cuda.py
```

## Usage

### Run Training

**Windows:**
```powershell
.\run_training.ps1
# or
run_training.bat
```

**Manual:**
```powershell
# Activate virtual environment
venv\Scripts\Activate.ps1

# Run training
python distill.py
```

## Configuration

Edit `distill.py` to modify:

- **Dataset size**: Change `num_prompts` (default: 20,000)
- **Batch size**: Adjust `per_device_train_batch_size` and `gradient_accumulation_steps`
- **Learning rate**: Modify `learning_rate` (default: 2e-4)
- **Training epochs**: Change `num_train_epochs` (default: 1)

## Training Configuration

```python
CONFIG = {
    "num_prompts": 20000,
    "training_config": {
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "num_train_epochs": 1,
        "fp16": True,
        "gradient_checkpointing": True,
    },
    "gkd_config": {
        "lmbda": 0.7,
        "beta": 0.5,
        "seq_kd": True
    }
}
```

## Output

- **Model**: Saved to `./student_model_gkd/`
- **Tokenizer**: Saved alongside the model
- **Training logs**: Displayed in console

## GPU Requirements

- **Minimum**: 8GB VRAM (RTX 3060 Ti / similar)
- **Recommended**: 16GB+ VRAM for larger batch sizes
- **Memory Usage**: ~10-12GB for full fine-tuning

## Files

- `distill.py` - Main training script
- `requirements.txt` - Python dependencies
- `check_cuda.py` - CUDA verification script
- `setup_venv.bat` / `setup_venv.sh` - Setup scripts
- `run_training.bat` / `run_training.ps1` - Training scripts

## Notes

- The script automatically checks for GPU availability
- Training will fail if CUDA is not available
- Model checkpoints are saved after each epoch
- Test prompts are evaluated after training

## License

See repository for license information.

## Acknowledgments

- OpenOrca dataset: https://huggingface.co/datasets/Open-Orca/OpenOrca
- TRL library: https://github.com/huggingface/trl
- Llama models: Meta AI

