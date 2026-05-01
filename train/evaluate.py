import os
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader


def evaluate_model(model_path: str = "models/empathy-model") -> None:
    """
    Evaluates the fine-tuned model on the validation set.
    
    Args:
        model_path: Path to the fine-tuned model
    """
    print("="*60)
    print("📊 Model Evaluation")
    print("="*60)
    
    # Load tokenizer and model
    print(f"\n📦 Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    print(f"✅ Model loaded successfully")
    
    # Load validation data
    print(f"\n📚 Loading validation dataset...")
    dataset = load_dataset(
        "json",
        data_files={"validation": "data/processed/validation.jsonl"},
    )
    
    # Tokenize
    def tokenize_function(batch):
        outputs = tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
            padding=False,
        )
        return outputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing validation set",
    )
    
    # Create trainer for evaluation
    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        per_device_eval_batch_size=4,
        logging_dir="logs",
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset["validation"],
    )
    
    # Evaluate
    print("\n🔍 Running evaluation...")
    eval_results = trainer.evaluate()
    
    # Display results
    print(f"\n📊 Evaluation Results:")
    print(f"   Eval Loss: {eval_results['eval_loss']:.4f}")
    
    perplexity = math.exp(eval_results["eval_loss"])
    print(f"   Perplexity: {perplexity:.2f}")
    print(f"""
   📌 Perplexity Interpretation:
      < 20  = Excellent (model is very confident)
      20-50 = Good for a fine-tuned small model
      > 100 = Model hasn't learned well
    """)


if __name__ == "__main__":
    evaluate_model()