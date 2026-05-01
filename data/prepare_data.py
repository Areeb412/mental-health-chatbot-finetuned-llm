import os
import json
from datasets import load_dataset
from tqdm import tqdm


# ─────────────────────────────────────────────────────────
# CONSTANTS — Change these if you want different splits
# ─────────────────────────────────────────────────────────
OUTPUT_DIR = "data/processed"
DATASET_NAME = "empathetic_dialogues"

# Special tokens we use to structure conversations
PERSON_TOKEN = "Person:"
SUPPORTER_TOKEN = "Supporter:"
END_TOKEN = "<|endoftext|>"


def format_conversation(row: dict) -> str | None:
    """
    Converts a single dataset row into a training string.
    
    The EmpatheticDialogues dataset has alternating utterances. We extract
    (prompt, utterance) pairs where:
      - prompt   = what the person struggling said
      - utterance = the empathetic listener's response
    
    Args:
        row: A dictionary with keys: prompt, utterance, context, etc.
    
    Returns:
        A formatted string like:
        "Person: I'm so stressed about exams. Supporter: I understand that feeling..."
        or None if data is invalid.
    
    Why this format?
        GPT-style models learn patterns in text. By consistently using
        "Person:" and "Supporter:" prefixes, the model learns that after
        "Supporter:" it should generate an empathetic response.
    """
    prompt = str(row.get("prompt", "")).strip()
    utterance = str(row.get("utterance", "")).strip()

    # Skip empty or very short exchanges — they won't teach the model much
    if len(prompt) < 5 or len(utterance) < 5:
        return None

    # Build the training string
    # The <|endoftext|> token tells the model where the conversation ends
    formatted = f"{PERSON_TOKEN} {prompt} {SUPPORTER_TOKEN} {utterance} {END_TOKEN}"
    return formatted


def prepare_dataset() -> None:
    """
    Main function: downloads, formats, and saves the dataset.
    
    We save as JSONL (JSON Lines) — one JSON object per line.
    This is the industry standard for large NLP datasets because:
    1. You can stream it line-by-line without loading everything into RAM
    2. It's human-readable
    3. Hugging Face's datasets library can load it natively
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("📥 Downloading EmpatheticDialogues from Hugging Face Hub...")

    # load_dataset downloads automatically and caches in ~/.cache/huggingface/
    # splits: "train", "validation", "test"
    dataset = load_dataset(DATASET_NAME, trust_remote_code=True)

    for split_name in ["train", "validation", "test"]:
        print(f"\n🔄 Processing '{split_name}' split...")
        split_data = dataset[split_name]
        
        formatted_examples = []
        skipped = 0

        for row in tqdm(split_data, desc=f"Formatting {split_name}"):
            result = format_conversation(row)
            if result is not None:
                formatted_examples.append({"text": result})
            else:
                skipped += 1

        # Write to JSONL file
        output_path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for example in formatted_examples:
                # json.dumps converts dict to JSON string, \n separates lines
                f.write(json.dumps(example) + "\n")

        print(f"✅ Saved {len(formatted_examples)} examples ({skipped} skipped) → {output_path}")

    print("\n🎉 Dataset preparation complete!")
    print(f"📁 Processed files saved in: {OUTPUT_DIR}/")
    print("\nSample training example:")
    print("-" * 60)
    print(formatted_examples[0]["text"])
    print("-" * 60)


if __name__ == "__main__":
    prepare_dataset()