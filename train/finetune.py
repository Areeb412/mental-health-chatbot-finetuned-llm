import os
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,          # Loads tokenizer matching any HF model
    AutoModelForCausalLM,   # Loads causal LM (GPT-style) model
    TrainingArguments,      # Configuration for the Trainer
    Trainer,                # The training engine
    DataCollatorForLanguageModeling,  # Handles batch preparation
    EarlyStoppingCallback,  # Stops training when model stops improving
)
import torch


# ─────────────────────────────────────────────────────────
# CONFIGURATION — Adjust these for your hardware
# ─────────────────────────────────────────────────────────

# The pre-trained model we start from
# "distilgpt2" is lighter than "gpt2" — good for laptops
BASE_MODEL_NAME = "distilgpt2"

# Where to save the fine-tuned model
OUTPUT_MODEL_DIR = "models/empathy-model"

# Where our processed data lives
DATA_DIR = "data/processed"

# Training hyperparameters (tuned for CPU training — small and fast)
MAX_LENGTH = 256          # Max tokens per training example (truncate longer ones)
BATCH_SIZE = 4            # Number of examples processed together — higher = more RAM
GRADIENT_ACCUMULATION = 4 # Simulate larger batch by accumulating gradients
NUM_EPOCHS = 3            # How many times to see the full dataset
LEARNING_RATE = 5e-5      # Size of each weight update step
WARMUP_STEPS = 100        # Gradually increase LR for first N steps (stabilizes training)
SAVE_STEPS = 500          # Save checkpoint every N steps
EVAL_STEPS = 500          # Evaluate on validation set every N steps
LOGGING_STEPS = 50        # Log training metrics every N steps


def load_and_tokenize_dataset(tokenizer):
    """
    Loads the JSONL dataset and converts text → token IDs.
    
    Tokenization Example:
        Input text:  "Person: I feel sad"
        Token IDs:   [15439, 25, 314, 1254, 6507]
        (Each number is an index into the model's 50,257-word vocabulary)
    
    Args:
        tokenizer: The loaded tokenizer object
    
    Returns:
        A tokenized HuggingFace Dataset ready for training
    """
    print("📚 Loading dataset from JSONL files...")
    
    # load_dataset can read JSONL directly — specify the 'text' field
    raw_dataset = load_dataset(
        "json",
        data_files={
            "train": f"{DATA_DIR}/train.jsonl",
            "validation": f"{DATA_DIR}/validation.jsonl",
        },
    )

    def tokenize_function(batch):
        """
        Tokenizes a batch of examples.
        
        truncation=True: If text is longer than MAX_LENGTH tokens, cut it off
        padding=False:   Don't pad yet — the DataCollator handles that per batch
        
        For causal LM, we DON'T manually set labels here.
        The DataCollatorForLanguageModeling will automatically create labels = input_ids
        and handle the -100 masking for padding tokens.
        """
        outputs = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        return outputs

    print("🔢 Tokenizing dataset (this may take a minute)...")
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,           # Process in batches of 1000 for speed
        remove_columns=["text"],  # Remove raw text — we only need token IDs
        desc="Tokenizing",
    )
    
    return tokenized_dataset


def train():
    """
    Main training function. Orchestrates the entire fine-tuning process.
    """
    print("=" * 60)
    print("🧠 Mental Health Chatbot — Fine-Tuning Pipeline")
    print("=" * 60)

    # ── STEP 1: Check hardware ────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Training device: {device.upper()}")
    if device == "cpu":
        print("⚠️  Training on CPU — this will take ~30-60 minutes for 3 epochs.")
        print("   For GPU training, use Google Colab (free) or reduce NUM_EPOCHS.")

    # ── STEP 2: Load tokenizer ───────────────────────────────
    print(f"\n📦 Loading tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    # GPT2-family models don't have a padding token by default
    # We add the end-of-text token as the pad token — this is the standard fix
    # Without this, batching would fail because sequences can't be padded
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("   ℹ️  Set pad_token = eos_token (GPT2 standard fix)")

    # ── STEP 3: Load base model ──────────────────────────────
    print(f"\n🤖 Loading base model: {BASE_MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    
    # Resize embedding layer if we added new special tokens
    # (In this project we don't add new tokens, but this is good practice)
    model.resize_token_embeddings(len(tokenizer))
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # ── STEP 4: Tokenize dataset ─────────────────────────────
    tokenized_dataset = load_and_tokenize_dataset(tokenizer)
    print(f"\n✅ Training examples: {len(tokenized_dataset['train']):,}")
    print(f"✅ Validation examples: {len(tokenized_dataset['validation']):,}")

    # ── STEP 5: Data Collator ────────────────────────────────
    # The DataCollator is responsible for building each training batch.
    # It pads all sequences in a batch to the same length.
    # mlm=False means we're doing Causal LM (not Masked LM like BERT)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-style = next token prediction, not masked token prediction
    )

    # ── STEP 6: Training arguments ───────────────────────────
    # TrainingArguments is a configuration class — all training settings live here
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    training_args = TrainingArguments(
        # Where to save checkpoints during training
        output_dir="models/checkpoints",
        
        # How many times to iterate over the full training dataset
        num_train_epochs=NUM_EPOCHS,
        
        # Batch size per device (GPU or CPU)
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        
        # Gradient accumulation: accumulate gradients over N steps before updating
        # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION = 4 * 4 = 16
        # This simulates a larger batch without needing more RAM
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        
        # Learning rate: how fast the model learns
        # Too high → training explodes / oscillates
        # Too low  → training is very slow
        # 5e-5 is the standard starting point for fine-tuning
        learning_rate=LEARNING_RATE,
        
        # Warmup: gradually increase LR from 0 to LEARNING_RATE over first N steps
        # This prevents large gradient updates at the start which can destabilize training
        warmup_steps=WARMUP_STEPS,
        
        # Evaluation strategy: evaluate on validation set every N steps
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        
        # Save checkpoints every N steps
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        
        # Keep only the 2 best checkpoints (saves disk space)
        save_total_limit=2,
        
        # Load the best model (not last) when training ends
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Use validation loss to decide "best"
        greater_is_better=False,            # Lower loss = better
        
        # Logging
        logging_steps=LOGGING_STEPS,
        logging_dir="logs",
        report_to="none",    # Don't send to W&B or TensorBoard (simplest setup)
        
        # FP16: half-precision training (faster on GPU, ignored on CPU)
        fp16=torch.cuda.is_available(),
        
        # Disable progress bar for cleaner output in production scripts
        disable_tqdm=False,
    )

    # ── STEP 7: Initialize Trainer ───────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        callbacks=[
            # Stop training if validation loss doesn't improve for 3 evaluations
            EarlyStoppingCallback(early_stopping_patience=3)
        ],
    )

    # ── STEP 8: TRAIN! ────────────────────────────────────────
    print("\n🚀 Starting fine-tuning...")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print("-" * 60)
    
    trainer.train()

    # ── STEP 9: Save the final model ─────────────────────────
    print(f"\n💾 Saving fine-tuned model to: {OUTPUT_MODEL_DIR}")
    trainer.save_model(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print("✅ Model saved successfully!")
    
    # ── STEP 10: Quick evaluation ─────────────────────────────
    print("\n📊 Final evaluation on validation set:")
    eval_results = trainer.evaluate()
    
    import math
    perplexity = math.exp(eval_results["eval_loss"])
    print(f"   Eval Loss: {eval_results['eval_loss']:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")
    print("""
   📌 Perplexity Interpretation:
      < 20  = Excellent (model is very confident)
      20-50 = Good for a fine-tuned small model
      > 100 = Model hasn't learned well (try more epochs)
    """)
    print("🎉 Fine-tuning complete! Run app.py to start the chatbot.")


if __name__ == "__main__":
    train()