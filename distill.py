from __future__ import annotations

import os
import torch
import warnings
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from datasets import load_dataset, Dataset
from trl import GKDTrainer, GKDConfig

# Set environment variables before importing tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================================
# Configuration
# ============================================================================
CONFIG: dict = {
    "teacher_model_id": "Qwen/Qwen2.5-7B-Instruct",
    "student_model_id": "Qwen/Qwen2.5-3B-Instruct",
    "dataset_file": "TeleQnA.json",
    "num_prompts": None,  # None means use entire dataset
    "gkd_config": {"lmbda": 0.7, "beta": 0.5, "seq_kd": True},
    "training_config": {
        "output_dir": "./student_model_gkd_Without_peft",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "num_train_epochs": 2,
        "logging_steps": 25,
        "report_to": "none",
        "save_strategy": "epoch",
        "fp16": False,  # Disabled to avoid AMP compatibility issues
        "bf16": False,  # Disabled to avoid AMP compatibility issues
        "gradient_checkpointing": True,
        "dataloader_pin_memory": True,  # Pin memory for faster GPU transfer
        "dataloader_num_workers": 4,  # Parallel data loading
    },
}

# ============================================================================
# GPU Verification
# ============================================================================
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available! GPU is required for training.")
    
device: torch.device = torch.device("cuda")
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# Load Dataset
# ============================================================================
import pandas as pd
import json

# Load JSON dataset
json_file = "TeleQnA.json"
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract prompts from the JSON
# Format: "Question: [question]\nOptions: [options]\nAnswer: [answer]"
prompts_list: list[str] = []
for key, item in data.items():
    if isinstance(item, dict):
        question = item.get("question", "")
        answer = item.get("answer", "")
        explanation = item.get("explanation", "")
        
        # Create a formatted prompt combining question and answer
        if question and answer:
            prompt = f"Q: {question}\nA: {answer}\nExplanation: {explanation}"
            prompts_list.append(prompt)

# Use entire dataset or limit if num_prompts is specified
if CONFIG["num_prompts"] is not None:
    prompts_list = prompts_list[:CONFIG["num_prompts"]]

print(f"Loaded {len(prompts_list)} prompts from dataset")

# ============================================================================
# Tokenizer
# ============================================================================
tokenizer = AutoTokenizer.from_pretrained(CONFIG["student_model_id"])
tokenizer.pad_token = tokenizer.eos_token

def format_data_for_gkd(example: dict) -> dict:
    """Format data for GKD training."""
    messages: list[dict] = [{"role": "user", "content": example["prompt"]}]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    }

hf_dataset = Dataset.from_dict({"prompt": prompts_list})
formatted_dataset = hf_dataset.map(format_data_for_gkd)

# ============================================================================
# Load Teacher Model
# ============================================================================
print("Loading teacher model...")
teacher_model = AutoModelForCausalLM.from_pretrained(
    CONFIG["teacher_model_id"],
    dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
teacher_model.eval()  # Set to eval mode
print("Teacher model loaded successfully")

# ============================================================================
# Load Student Model
# ============================================================================
print("Loading student model...")
student_model = AutoModelForCausalLM.from_pretrained(
    CONFIG["student_model_id"],
    dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
student_model.train()  # Set to training mode
print("Student model loaded successfully")

# ============================================================================
# Align Vocabularies
# ============================================================================
print(f"Before alignment: Teacher vocab_size={teacher_model.config.vocab_size}, Student vocab_size={student_model.config.vocab_size}")

# Resize student model token embeddings to match teacher
if student_model.config.vocab_size != teacher_model.config.vocab_size:
    student_model.resize_token_embeddings(
        teacher_model.config.vocab_size,
        pad_to_multiple_of=64,
        mean_resizing=False  # Use simple truncation/padding instead of mean resizing
    )
    print(f"Resized student model token embeddings")

print(f"After alignment: Teacher vocab_size={teacher_model.config.vocab_size}, Student vocab_size={student_model.config.vocab_size}")

# ============================================================================
# Tokenize Dataset
# ============================================================================
def tokenize_function(examples: dict) -> dict:
    """Tokenize examples for training."""
    return tokenizer(examples["text"], truncation=True, max_length=256)

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

# ============================================================================
# Training Configuration
# ============================================================================
# Set use_cache=False when using gradient_checkpointing
if CONFIG["training_config"].get("gradient_checkpointing"):
    student_model.config.use_cache = False
    teacher_model.config.use_cache = False

training_args = GKDConfig(
	**CONFIG["training_config"], **CONFIG["gkd_config"], remove_unused_columns=False
)

# ============================================================================
# Initialize Trainer
# ============================================================================
trainer = GKDTrainer(
	model=student_model,
	teacher_model=teacher_model,
	args=training_args,
	train_dataset=tokenized_dataset,
	processing_class=tokenizer,
)

# ============================================================================
# Monkey-patch to fix dtype in generation (if needed)
# ============================================================================
original_generate = student_model.generate

def patched_generate(*args, **kwargs) -> torch.Tensor:
    """Patched generate function with proper device placement."""
    # Ensure input tensors are on the correct device
    if "input_ids" in kwargs:
        kwargs["input_ids"] = kwargs["input_ids"].to(student_model.device)
    if "attention_mask" in kwargs:
        kwargs["attention_mask"] = kwargs["attention_mask"].to(student_model.device)
    # Call generate without autocast since we're using fp32
    return original_generate(*args, **kwargs)

student_model.generate = patched_generate

# ============================================================================
# Train
# ============================================================================
trainer.train()

# ============================================================================
# Save
# ============================================================================
model_path = "./student_model_gkd"
student_model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# ============================================================================
# Test
# ============================================================================
test_prompts: list[str] = [
    "What is artificial intelligence?",
    "Explain quantum computing in simple terms.",
]

distilled_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    dtype="auto",
    trust_remote_code=True
)
distilled_model.eval()

for prompt in test_prompts:
    messages: list[dict] = [{"role": "user", "content": prompt}]
    input_text: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = distilled_model.generate(
        **inputs, max_new_tokens=256, do_sample=True, temperature=0.7
    )
    response: str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer: str = response.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]
    print(f"Q: {prompt}\nA: {answer}\n{'-'*70}")
