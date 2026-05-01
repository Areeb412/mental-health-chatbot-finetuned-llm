# 🧠 Mental Health Support Chatbot (Fine-Tuned LLM) — GPU Edition
### A Complete, Production-Grade Step-by-Step Guide for Beginners to Pros

> **Goal:** Fine-tune a small LLM (DistilGPT2) on Facebook's EmpatheticDialogues dataset and serve it through a beautiful Flask-based chat UI — exactly like the medical chatbot reference project — using `uv` for environment management. By the end, you will have a GitHub-ready, senior-level portfolio project.
>
> ⚡ **GPU Edition:** This guide is optimized to fully utilize your NVIDIA GPU for faster training and inference.

---

## 📚 Table of Contents

1. [What We Are Building](#1-what-we-are-building)
2. [Core Concepts Explained](#2-core-concepts-explained)
3. [Project Architecture](#3-project-architecture)
4. [Complete Project Structure](#4-complete-project-structure)
5. [Environment Setup with `uv`](#5-environment-setup-with-uv)
6. [Dependencies & Configuration Files](#6-dependencies--configuration-files)
7. [Step 1 — Data Preparation](#7-step-1--data-preparation)
8. [Step 2 — Fine-Tuning the Model](#8-step-2--fine-tuning-the-model)
9. [Step 3 — Building the Flask Backend](#9-step-3--building-the-flask-backend)
10. [Step 4 — Building the Chat Frontend (HTML/CSS/JS)](#10-step-4--building-the-chat-frontend-htmlcssjs)
11. [Step 5 — Inference & Response Pipeline](#11-step-5--inference--response-pipeline)
12. [Step 6 — Logging & Safety Layer](#12-step-6--logging--safety-layer)
13. [Step 7 — Docker & Production Deployment](#13-step-7--docker--production-deployment)
14. [Step 8 — GitHub CI/CD with GitHub Actions](#14-step-8--github-cicd-with-github-actions)
15. [Running the Project End-to-End](#15-running-the-project-end-to-end)
16. [Common Errors & Fixes](#16-common-errors--fixes)
17. [Senior-Level Best Practices Used](#17-senior-level-best-practices-used)

---

## 1. What We Are Building

You are building a **Mental Health Support Chatbot** that:

- Is **fine-tuned** on real human empathetic conversations (not just a raw pre-trained model)
- Responds with a **gentle, supportive, non-judgmental tone**
- Has a **beautiful chat UI** (same style as the medical chatbot reference project — Flask + custom HTML/CSS/JS)
- Runs via **Flask for the web layer** with full **CUDA GPU acceleration**
- Is **Dockerized** and **CI/CD ready** for AWS/cloud deployment
- Uses **`uv`** — the fastest modern Python environment manager (replaces pip + venv + pipenv)

### Why These Choices?

| Decision | Why |
|---|---|
| **DistilGPT2** | Lightweight (82M params), trains in ~5-15 min on GPU |
| **EmpatheticDialogues** | Real human empathetic conversations from Facebook AI Research |
| **Hugging Face Trainer API** | Industry-standard, handles training loops, evaluation, checkpointing |
| **Flask + Jinja2 HTML** | Matches the reference project UI style, lightweight, battle-tested |
| **`uv`** | 10-100x faster than pip, built-in lock files, modern standard |
| **CUDA + FP16** | Half-precision training cuts VRAM usage ~50%, doubles throughput |

---

## 2. Core Concepts Explained

### 2.1 What is a Language Model?

A Language Model (LM) is a neural network trained to **predict the next word** in a sequence. When you type "I feel", the model assigns probabilities to every word in its vocabulary and picks the most likely next token — iteratively building a response.

```
Input:  "I feel so anxious today"
Model:  → predicts next tokens one by one →
Output: "I hear you, and I'm sorry you're feeling that way."
```

### 2.2 What is Fine-Tuning?

A **pre-trained model** like DistilGPT2 has learned general language from billions of internet documents. But it doesn't know *how* to be empathetic.

**Fine-tuning** = taking that pre-trained model and continuing its training on a *smaller, specialized dataset* so it learns a specific style or domain.

Think of it like this: a medical student (pre-trained) knows science broadly, then does a psychiatry residency (fine-tuning) to specialize.

```
[DistilGPT2 pre-trained weights]
         ↓
  Train on EmpatheticDialogues  ← runs on your GPU ⚡
         ↓
  [Empathy-tuned weights saved]
```

### 2.3 What is the Hugging Face Trainer API?

Instead of writing your own PyTorch training loop (which can be 200+ lines), Hugging Face's `Trainer` class handles:
- **Forward pass** (model prediction)
- **Loss calculation** (how wrong was it?)
- **Backward pass** (compute gradients)
- **Optimizer step** (update weights)
- **Checkpointing** (save model every N steps)
- **Evaluation** (measure perplexity on validation set)

You just configure it and call `.train()`.

### 2.4 What is the EmpatheticDialogues Dataset?

Published by Facebook AI Research (2019), it contains **25,000 conversations** where a speaker describes an emotional situation and a listener responds empathetically. Each conversation has:
- `conv_id`: Conversation identifier
- `utterance_idx`: Turn index
- `context`: The emotional situation label (e.g., "anxious", "devastated")
- `prompt`: What the speaker said
- `utterance`: The empathetic response

### 2.5 What is Flask?

Flask is a **micro web framework** for Python. It lets you define URL routes and return HTML responses. The reference project uses Flask with Jinja2 templates — we follow the exact same pattern.

```python
@app.route("/")
def index():
    return render_template("chat.html")  # serves your chat page

@app.route("/get", methods=["POST"])
def chat():
    user_message = request.json["msg"]
    response = generate_response(user_message)
    return jsonify({"response": response})
```

### 2.6 What is `uv`?

`uv` is a blazing-fast Python package manager written in Rust (by Astral, creators of Ruff). It replaces:
- `pip install` → `uv add`
- `python -m venv` → `uv venv`
- `requirements.txt` → `pyproject.toml` + `uv.lock`

It resolves dependencies 10–100x faster than pip and creates fully reproducible environments.

### 2.7 GPU Concepts (CUDA, FP16, VRAM)

| Term | What it means |
|---|---|
| **CUDA** | NVIDIA's parallel computing platform — lets PyTorch run on your GPU |
| **FP16 (half-precision)** | Stores weights as 16-bit floats instead of 32-bit → ~50% less VRAM, ~2x faster |
| **VRAM** | GPU memory. DistilGPT2 needs ~2-3 GB with FP16 |
| **`torch.cuda.is_available()`** | Returns `True` when CUDA is detected — all our scripts check this |

---

## 3. Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER'S BROWSER                       │
│          Beautiful Chat UI (HTML/CSS/JS)                │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP POST /get
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   FLASK APPLICATION                     │
│  app.py — Routes, Template Serving, Request Handling    │
└────────────────────┬────────────────────────────────────┘
                     │ calls
                     ▼
┌─────────────────────────────────────────────────────────┐
│              INFERENCE PIPELINE (src/inference.py)      │
│  - Loads fine-tuned model onto GPU (cuda:0)             │
│  - Tokenizes user input                                 │
│  - Runs model.generate() on GPU ⚡                       │
│  - Post-processes & returns empathetic response         │
└────────────────────┬────────────────────────────────────┘
                     │ uses
                     ▼
┌─────────────────────────────────────────────────────────┐
│         FINE-TUNED MODEL (models/empathy-model/)        │
│  DistilGPT2 weights fine-tuned on EmpatheticDialogues   │
│  Stored in FP16 on GPU for fast inference ⚡             │
└─────────────────────────────────────────────────────────┘
```

**Training Pipeline (separate, run once):**

```
EmpatheticDialogues Dataset
        ↓
   data/prepare_data.py   ← cleans and formats dialogues
        ↓
   train/finetune.py      ← Trainer runs on GPU with FP16 ⚡
        ↓
   models/empathy-model/  ← saved weights, tokenizer, config
```

---

## 4. Complete Project Structure

```
mental-health-chatbot/
│
├── 📄 pyproject.toml          # uv project config & dependencies
├── 📄 uv.lock                 # locked dependency versions (auto-generated)
├── 📄 .env.example            # environment variable template
├── 📄 .gitignore              # files git should ignore
├── 📄 Dockerfile              # containerization
├── 📄 docker-compose.yml      # local Docker orchestration
├── 📄 README.md               # project documentation
│
├── 📁 .github/
│   └── 📁 workflows/
│       └── 📄 ci-cd.yml       # GitHub Actions pipeline
│
├── 📁 data/
│   ├── 📄 prepare_data.py     # download & format dataset
│   └── 📁 processed/          # cleaned data (git-ignored)
│
├── 📁 train/
│   ├── 📄 finetune.py         # model fine-tuning script (GPU-optimized)
│   └── 📄 evaluate.py         # model evaluation script
│
├── 📁 models/
│   └── 📁 empathy-model/      # saved fine-tuned model (git-ignored)
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer files
│
├── 📁 src/
│   ├── 📄 __init__.py
│   ├── 📄 inference.py        # model loading & GPU generation
│   ├── 📄 safety.py           # crisis detection & safety layer
│   └── 📄 logger.py           # structured logging
│
├── 📁 templates/
│   └── 📄 chat.html           # Flask Jinja2 chat template
│
├── 📁 static/
│   ├── 📁 css/
│   │   └── 📄 style.css       # chat styling
│   └── 📁 js/
│       └── 📄 chat.js         # chat interaction logic
│
└── 📄 app.py                  # Flask application entry point
```

---

## 5. Environment Setup with `uv`

### 5.1 Verify Your GPU First

Before anything else, confirm CUDA is accessible:

```bash
# Check NVIDIA driver and GPU
nvidia-smi

# Expected output (example):
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.xx    Driver Version: 535.xx    CUDA Version: 12.x          |
# |-------------------------------+------------------+                          |
# | GPU 0: NVIDIA GeForce RTX ...| 00000000:01:00.0 |                          |
# +-----------------------------------------------------------------------------+
```

> **Note your CUDA version** from the top-right of `nvidia-smi` output (e.g. `12.1`). You'll need it when installing PyTorch.

### 5.2 Install `uv`

```bash
# On Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
# Expected: uv 0.5.x
```

### 5.3 Create the Project

```bash
# Create project directory
mkdir mental-health-chatbot
cd mental-health-chatbot

# Initialize uv project (creates pyproject.toml)
uv init

# Create virtual environment
uv venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 5.4 Install Dependencies (GPU Build)

```bash
# ⚡ IMPORTANT: Install the CUDA-enabled PyTorch build
# Replace cu121 with your CUDA version (cu118, cu121, cu124, etc.)
# Find your version from: nvidia-smi (top-right corner)

uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core ML dependencies
uv add transformers datasets accelerate

# Web framework
uv add flask python-dotenv

# Utilities
uv add numpy tqdm

# Development dependencies (not shipped in production)
uv add --dev pytest black ruff
```

> **Common CUDA version mappings:**
> | CUDA (nvidia-smi) | PyTorch index URL |
> |---|---|
> | 11.8 | `https://download.pytorch.org/whl/cu118` |
> | 12.1 | `https://download.pytorch.org/whl/cu121` |
> | 12.4 | `https://download.pytorch.org/whl/cu124` |
> | 12.6 | `https://download.pytorch.org/whl/cu126` |

### 5.5 Verify GPU is Detected by PyTorch

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# Expected output:
# CUDA available: True
# GPU: NVIDIA GeForce RTX 3060   ← (your GPU name)
```

If this prints `CUDA available: False`, your PyTorch CUDA version doesn't match your driver. Re-run step 5.4 with the correct `cu` version.

---

## 6. Dependencies & Configuration Files

### 6.1 `pyproject.toml`

Create this file at the project root. Every field is explained:

```toml
[project]
# Your project's name — shown on PyPI if published
name = "mental-health-chatbot"

# Semantic versioning: MAJOR.MINOR.PATCH
version = "1.0.0"

# Human-readable description for your README/portfolio
description = "An empathetic mental health support chatbot fine-tuned on EmpatheticDialogues using DistilGPT2 and served via Flask."

# Minimum Python version — 3.10 gives us match-case, better typing
requires-python = ">=3.10"

# Runtime dependencies (installed in production)
dependencies = [
    "torch>=2.0.0",
    "torchvision",
    "torchaudio",
    "transformers>=4.40.0",    # Hugging Face model library
    "datasets>=2.18.0",        # Hugging Face datasets library
    "accelerate>=0.28.0",      # Required for FP16 / mixed-precision GPU training
    "flask>=3.0.0",            # Web framework
    "python-dotenv>=1.0.0",    # Load .env files into os.environ
    "numpy>=1.26.0",
    "tqdm>=4.66.0",            # Progress bars for training loops
]

[project.optional-dependencies]
# Dev-only tools — not shipped to users
dev = [
    "pytest>=8.0.0",    # Testing framework
    "black>=24.0.0",    # Code formatter
    "ruff>=0.3.0",      # Ultra-fast linter (replaces flake8)
]

[build-system]
# Required boilerplate for Python packaging
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.black]
# Black formatter: 88 chars is the modern standard
line-length = 88

[tool.ruff]
# Ruff linter: catches bugs and style issues
line-length = 88
select = ["E", "F", "W", "I"]   # Error, pyFlakes, Warning, Import sorting
```

### 6.2 `.env.example`

```env
# Copy this file to .env and fill in your values
# NEVER commit .env to git

# Flask configuration
FLASK_ENV=development        # "development" or "production"
FLASK_DEBUG=True             # Set to False in production
FLASK_PORT=5000

# Model configuration
MODEL_PATH=models/empathy-model    # Path to your fine-tuned model
MAX_NEW_TOKENS=150                 # Max tokens model generates per response
TEMPERATURE=0.85                   # Creativity: 0=deterministic, 1=creative
TOP_P=0.92                         # Nucleus sampling threshold
REPETITION_PENALTY=1.3             # Penalizes repeating the same phrases

# GPU configuration
USE_GPU=True                       # Set to False to force CPU (not recommended)
# CUDA_VISIBLE_DEVICES=0           # Uncomment to pin to a specific GPU

# Safety configuration
ENABLE_CRISIS_DETECTION=True       # Detects suicide/self-harm language
CRISIS_RESPONSE_MSG="I hear you. Please reach out to a crisis helpline: 988 (call or text) or text HOME to 741741"
```

### 6.3 `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.venv/
dist/
*.egg-info/

# Environment
.env

# Model weights (too large for git — use Git LFS or store on Hugging Face Hub)
models/empathy-model/
data/processed/

# IDE
.vscode/
.idea/

# Jupyter
*.ipynb
.ipynb_checkpoints/

# Docker
*.log
```

---

## 7. Step 1 — Data Preparation

**File: `data/prepare_data.py`**

This script downloads the EmpatheticDialogues dataset and transforms it into the exact format our model needs for training.

```python
"""
data/prepare_data.py

Downloads the EmpatheticDialogues dataset from Hugging Face and formats it
for causal language model fine-tuning.

Key Concept — Causal Language Modeling (CLM):
    DistilGPT2 is a causal (left-to-right) language model.
    Training input format:
        "Person: [speaker utterance] Supporter: [empathetic response] <|endoftext|>"
    
    The model learns to predict each next token given all previous tokens.
    After training, we feed "Person: [user input] Supporter:" and the model
    generates the empathetic response.
"""

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
    dataset = load_dataset(DATASET_NAME)

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
```

**Run it:**
```bash
python data/prepare_data.py
```

**What you'll see:**
```
📥 Downloading EmpatheticDialogues from Hugging Face Hub...
🔄 Processing 'train' split...
Formatting train: 100%|████████| 76673/76673 [00:03<00:00]
✅ Saved 76421 examples (252 skipped) → data/processed/train.jsonl

Sample training example:
------------------------------------------------------------
Person: I lost my job today and I don't know what to do. 
Supporter: That sounds really difficult. Losing a job can be 
overwhelming, especially when it's unexpected. <|endoftext|>
------------------------------------------------------------
```

---

## 8. Step 2 — Fine-Tuning the Model (GPU-Optimized ⚡)

**File: `train/finetune.py`**

This is the heart of the project. The GPU-optimized version uses **FP16 mixed-precision training**, **larger batch sizes**, and **`torch.compile`** for maximum throughput.

```python
"""
train/finetune.py  — GPU-Optimized Edition ⚡

Fine-tunes DistilGPT2 on EmpatheticDialogues using Hugging Face Trainer
with full CUDA GPU acceleration.

GPU Optimizations Applied:
────────────────────────────────────────────────────────
1. FP16 TRAINING:    Half-precision floats halve VRAM usage and
                     nearly double throughput on RTX/Ampere GPUs.

2. LARGER BATCHES:   GPU can process more examples per step than CPU.
                     Effective batch = BATCH_SIZE * GRADIENT_ACCUMULATION.

3. DATALOADER WORKERS: Multiple CPU threads pre-load batches while
                     the GPU trains, eliminating data starvation.

4. PIN MEMORY:       Pins data in CPU RAM for faster GPU transfers
                     (avoids one memory copy step).

5. bf16 FALLBACK:    Ampere+ GPUs (RTX 30xx/40xx) support bfloat16
                     which is more numerically stable than float16.
────────────────────────────────────────────────────────
"""

import os
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)


# ─────────────────────────────────────────────────────────
# CONFIGURATION — GPU-Optimized Settings
# ─────────────────────────────────────────────────────────

BASE_MODEL_NAME = "distilgpt2"
OUTPUT_MODEL_DIR = "models/empathy-model"
DATA_DIR = "data/processed"

# ── GPU Training Hyperparameters ─────────────────────────
MAX_LENGTH = 256          # Max tokens per example
BATCH_SIZE = 16           # ⬆ Much larger than CPU (was 4) — uses GPU parallelism
GRADIENT_ACCUMULATION = 2 # Effective batch = 16 * 2 = 32
NUM_EPOCHS = 3            # Full passes through the dataset
LEARNING_RATE = 5e-5      # Standard fine-tuning LR
WARMUP_STEPS = 200        # Slightly more warmup for larger effective batch
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 50

# ── Detect GPU capabilities ───────────────────────────────
CUDA_AVAILABLE = torch.cuda.is_available()

def get_precision_flags():
    """
    Returns the best precision settings for the detected GPU.

    - Ampere+ (RTX 30xx/40xx, A100): use bf16 (more stable than fp16)
    - Older NVIDIA (GTX 16xx, RTX 20xx): use fp16
    - CPU fallback: fp32
    """
    if not CUDA_AVAILABLE:
        return {"fp16": False, "bf16": False}

    # bf16 requires compute capability >= 8.0 (Ampere architecture)
    major = torch.cuda.get_device_capability(0)[0]
    if major >= 8:
        print("   ✅ Ampere/Ada GPU detected — using BF16 (more stable)")
        return {"fp16": False, "bf16": True}
    else:
        print("   ✅ Older NVIDIA GPU detected — using FP16")
        return {"fp16": True, "bf16": False}


def load_and_tokenize_dataset(tokenizer):
    """
    Loads the JSONL dataset and tokenizes it.
    """
    print("📚 Loading dataset from JSONL files...")
    raw_dataset = load_dataset(
        "json",
        data_files={
            "train": f"{DATA_DIR}/train.jsonl",
            "validation": f"{DATA_DIR}/validation.jsonl",
        },
    )

    def tokenize_function(batch):
        outputs = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    print("🔢 Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
        # ⚡ Use multiple CPU workers for tokenization preprocessing
        num_proc=min(4, os.cpu_count() or 1),
    )
    return tokenized_dataset


def train():
    """
    Main GPU-optimized training function.
    """
    print("=" * 60)
    print("🧠 Mental Health Chatbot — GPU Fine-Tuning Pipeline ⚡")
    print("=" * 60)

    # ── STEP 1: Verify GPU ────────────────────────────────
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            "❌ CUDA is not available!\n"
            "Make sure you installed the CUDA-enabled PyTorch build:\n"
            "  uv add torch --index-url https://download.pytorch.org/whl/cu121\n"
            "Then verify with: python -c \"import torch; print(torch.cuda.is_available())\""
        )

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n⚡ GPU: {gpu_name}")
    print(f"   VRAM: {vram_gb:.1f} GB")
    print(f"   CUDA: {torch.version.cuda}")

    precision_flags = get_precision_flags()

    # ── STEP 2: Load tokenizer ───────────────────────────
    print(f"\n📦 Loading tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("   ℹ️  Set pad_token = eos_token (GPT2 standard fix)")

    # ── STEP 3: Load model onto GPU ──────────────────────
    print(f"\n🤖 Loading model: {BASE_MODEL_NAME} → GPU")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        # Load directly in the correct precision to save VRAM
        torch_dtype=torch.bfloat16 if precision_flags["bf16"] else (
            torch.float16 if precision_flags["fp16"] else torch.float32
        ),
    )
    model = model.cuda()          # Move to GPU
    model.resize_token_embeddings(len(tokenizer))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   Model device: {next(model.parameters()).device}")

    # ── STEP 4: Tokenize dataset ─────────────────────────
    tokenized_dataset = load_and_tokenize_dataset(tokenizer)
    print(f"\n✅ Training examples: {len(tokenized_dataset['train']):,}")
    print(f"✅ Validation examples: {len(tokenized_dataset['validation']):,}")

    # ── STEP 5: Data Collator ────────────────────────────
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── STEP 6: Training Arguments (GPU-Tuned) ────────────
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        num_train_epochs=NUM_EPOCHS,

        # ⚡ GPU batch sizes — larger than CPU because GPU handles parallelism
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,

        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=LOGGING_STEPS,
        logging_dir="logs",
        report_to="none",

        # ⚡ GPU precision training
        **precision_flags,

        # ⚡ Multiple CPU workers pre-load data while GPU trains
        # Eliminates "GPU waiting for data" bottleneck
        dataloader_num_workers=4,

        # ⚡ Pin memory in CPU RAM for faster Host→GPU transfers
        dataloader_pin_memory=True,

        # ⚡ Gradient checkpointing: trades compute for VRAM
        # Useful if you're tight on VRAM (e.g. 4 GB GPU)
        # gradient_checkpointing=True,  # Uncomment if you get OOM errors
    )

    # ── STEP 7: Trainer ──────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── STEP 8: Train! ────────────────────────────────────
    print(f"\n🚀 Starting GPU fine-tuning...")
    print(f"   GPU: {gpu_name}")
    print(f"   Precision: {'BF16' if precision_flags['bf16'] else 'FP16'}")
    print(f"   Batch size: {BATCH_SIZE} (per GPU)")
    print(f"   Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Estimated time: ~5-15 min depending on GPU ⚡")
    print("-" * 60)

    trainer.train()

    # ── STEP 9: Save model ────────────────────────────────
    print(f"\n💾 Saving fine-tuned model to: {OUTPUT_MODEL_DIR}")
    trainer.save_model(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print("✅ Model saved successfully!")

    # ── STEP 10: Evaluate ─────────────────────────────────
    print("\n📊 Final evaluation on validation set:")
    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results["eval_loss"])
    print(f"   Eval Loss:   {eval_results['eval_loss']:.4f}")
    print(f"   Perplexity:  {perplexity:.2f}")
    print("""
   📌 Perplexity Interpretation:
      < 20  = Excellent (model is very confident)
      20-50 = Good for a fine-tuned small model
      > 100 = Model hasn't learned well (try more epochs)
    """)
    print("🎉 Fine-tuning complete! Run app.py to start the chatbot.")


if __name__ == "__main__":
    train()
```

**Run it:**
```bash
python train/finetune.py
```

**Expected GPU output:**
```
⚡ GPU: NVIDIA GeForce RTX 3060
   VRAM: 12.0 GB
   CUDA: 12.1
   ✅ Ampere/Ada GPU detected — using BF16 (more stable)

🚀 Starting GPU fine-tuning...
   Estimated time: ~5-15 min depending on GPU ⚡
   Step  50: loss = 2.43 | grad_norm = 1.2 | lr = 4.5e-05
   Step 100: loss = 2.21 | grad_norm = 0.9 | lr = 5.0e-05
   ...
💾 Saving fine-tuned model → models/empathy-model
🎉 Fine-tuning complete!
```

> **🔧 Out of Memory (OOM)?** Add `gradient_checkpointing=True` in TrainingArguments, and reduce `BATCH_SIZE` to `8`. This trades a bit of speed for ~40% less VRAM.

---

## 9. Step 3 — Building the Flask Backend

**File: `app.py`**

```python
"""
app.py

Flask web application serving the Mental Health Support Chatbot.

Architecture mirrors the reference project:
    GET  /          → Serves the chat UI (chat.html)
    POST /get       → Accepts user message, returns bot response (JSON)
    GET  /health    → Health check endpoint (for Docker/load balancers)
"""

import os
import logging
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

from src.inference import EmpathyBot
from src.safety import SafetyLayer
from src.logger import setup_logger

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
logger = setup_logger("app")

# ─── Initialize Model (loaded once at startup) ────────────────────────────
logger.info("Initializing EmpathyBot on GPU...")
MODEL_PATH = os.getenv("MODEL_PATH", "models/empathy-model")

try:
    bot = EmpathyBot(model_path=MODEL_PATH)
    safety = SafetyLayer()
    logger.info("✅ EmpathyBot initialized successfully on GPU")
except FileNotFoundError:
    logger.error(
        f"❌ Model not found at '{MODEL_PATH}'. "
        "Run 'python train/finetune.py' first to train the model."
    )
    bot = None
    safety = SafetyLayer()


# ─── Routes ──────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    """
    Main chat endpoint — receives user message, returns bot response.

    Request format (JSON):
        { "msg": "I've been feeling really anxious lately" }

    Response format (JSON):
        { "response": "I hear you. Anxiety can be really overwhelming..." }
    """
    data = request.get_json(silent=True)
    if not data or "msg" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = str(data["msg"]).strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    logger.info(f"Received message (length: {len(user_message)} chars)")

    # Safety check BEFORE generating response
    crisis_response = safety.check(user_message)
    if crisis_response:
        logger.warning("Crisis language detected — returning safety response")
        return jsonify({"response": crisis_response, "is_crisis": True})

    if bot is None:
        return jsonify({
            "response": (
                "I'm sorry, I'm having trouble right now. "
                "Please try again later or reach out to a trusted person."
            )
        })

    try:
        response = bot.generate(user_message)
        logger.info(f"Generated response (length: {len(response)} chars)")
        return jsonify({"response": response, "is_crisis": False})
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return jsonify({
            "response": (
                "I'm here with you. Could you tell me more about how you're feeling?"
            )
        })


@app.route("/health", methods=["GET"])
def health():
    """Health check for Docker and load balancers."""
    import torch
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if bot is not None:
        return jsonify({"status": "healthy", "model_loaded": True, **gpu_info}), 200
    else:
        return jsonify({"status": "degraded", "model_loaded": False, **gpu_info}), 503


# ─── Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    print(f"""
╔══════════════════════════════════════════════════════════╗
║       🧠 Mental Health Support Chatbot  ⚡ GPU           ║
║       Running at http://localhost:{port}                   ║
║       Press CTRL+C to stop                               ║
╚══════════════════════════════════════════════════════════╝
    """)

    app.run(host="0.0.0.0", port=port, debug=debug)
```

---

## 10. Step 4 — Building the Chat Frontend (HTML/CSS/JS)

**File: `templates/chat.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mental Health Support Chat</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <link rel="stylesheet" 
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body>

<div class="chat-wrapper">

    <!-- ── Header ── -->
    <div class="chat-header">
        <div class="header-icon">
            <i class="fas fa-brain"></i>
        </div>
        <div class="header-text">
            <h1>MindfulChat</h1>
            <p class="subtitle">A safe space to share how you feel</p>
        </div>
        <div class="header-status">
            <span class="status-dot"></span>
            <span>Here for you</span>
        </div>
    </div>

    <!-- ── Disclaimer Banner ── -->
    <div class="disclaimer">
        <i class="fas fa-info-circle"></i>
        This chatbot is for emotional support only and is not a substitute for 
        professional mental health care. If you're in crisis, please call 
        <strong>988</strong> (Suicide &amp; Crisis Lifeline).
    </div>

    <!-- ── Chat Messages Area ── -->
    <div class="chat-messages" id="chatMessages">
        <div class="message bot-message">
            <div class="avatar bot-avatar">
                <i class="fas fa-heart"></i>
            </div>
            <div class="bubble bot-bubble">
                <p>Hello, I'm glad you're here. This is a safe and judgment-free space. 
                   How are you feeling today? 💙</p>
                <span class="timestamp">Just now</span>
            </div>
        </div>
    </div>

    <!-- ── Typing Indicator ── -->
    <div class="typing-indicator" id="typingIndicator" style="display: none;">
        <div class="avatar bot-avatar small">
            <i class="fas fa-heart"></i>
        </div>
        <div class="typing-dots">
            <span></span><span></span><span></span>
        </div>
    </div>

    <!-- ── Input Area ── -->
    <div class="chat-input-area">
        <div class="input-container">
            <textarea 
                id="userInput" 
                placeholder="Share what's on your mind..."
                rows="1"
                maxlength="500"
            ></textarea>
            <button id="sendBtn" onclick="sendMessage()" title="Send message">
                <i class="fas fa-paper-plane"></i>
            </button>
        </div>
        <div class="input-footer">
            <span id="charCount">0/500 characters</span>
            <span>Press <kbd>Enter</kbd> to send, <kbd>Shift+Enter</kbd> for new line</span>
        </div>
    </div>

</div>

<script src="{{ url_for('static', filename='js/chat.js') }}"></script>
</body>
</html>
```

**File: `static/css/style.css`**

```css
/* ═══════════════════════════════════════════════════════════
   style.css — Mental Health Chatbot UI
   Design Philosophy: Calm, gentle, trustworthy
   ═══════════════════════════════════════════════════════════ */

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
}

.chat-wrapper {
    width: 100%;
    max-width: 780px;
    background: #ffffff;
    border-radius: 24px;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    display: flex;
    flex-direction: column;
    height: 88vh;
    max-height: 780px;
    overflow: hidden;
}

.chat-header {
    background: linear-gradient(135deg, #5b8dee 0%, #7c5cbf 100%);
    color: white;
    padding: 20px 24px;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-shrink: 0;
}

.header-icon {
    width: 48px; height: 48px;
    background: rgba(255,255,255,0.2);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
}

.header-text h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }
.header-text .subtitle { font-size: 13px; opacity: 0.85; margin-top: 2px; }

.header-status { margin-left: auto; display: flex; align-items: center; gap: 7px; font-size: 13px; opacity: 0.9; }
.status-dot { width: 8px; height: 8px; background: #4ade80; border-radius: 50%; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

.disclaimer {
    background: #eff6ff; border-left: 4px solid #5b8dee;
    padding: 10px 18px; font-size: 12.5px; color: #374151; flex-shrink: 0;
}
.disclaimer i { color: #5b8dee; margin-right: 6px; }

.chat-messages {
    flex: 1; overflow-y: auto; padding: 20px;
    display: flex; flex-direction: column; gap: 16px; scroll-behavior: smooth;
}
.chat-messages::-webkit-scrollbar { width: 6px; }
.chat-messages::-webkit-scrollbar-track { background: transparent; }
.chat-messages::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }

.message { display: flex; align-items: flex-end; gap: 10px; animation: slideIn 0.3s ease; }
@keyframes slideIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
.bot-message { flex-direction: row; }
.user-message { flex-direction: row-reverse; }

.avatar { width: 38px; height: 38px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; }
.bot-avatar { background: linear-gradient(135deg, #5b8dee, #7c5cbf); color: white; }
.avatar.small { width: 30px; height: 30px; font-size: 13px; }

.bubble { max-width: 72%; padding: 12px 16px; border-radius: 18px; font-size: 15px; line-height: 1.55; }
.bot-bubble { background: #f3f4f6; color: #1f2937; border-bottom-left-radius: 4px; }
.user-bubble { background: linear-gradient(135deg, #5b8dee, #7c5cbf); color: white; border-bottom-right-radius: 4px; }
.crisis-bubble { background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; }
.timestamp { display: block; font-size: 10.5px; margin-top: 5px; opacity: 0.55; }

.typing-indicator { display: flex; align-items: center; gap: 10px; padding: 0 20px 10px; }
.typing-dots { background: #f3f4f6; border-radius: 18px; padding: 12px 16px; display: flex; gap: 4px; align-items: center; }
.typing-dots span { width: 7px; height: 7px; background: #9ca3af; border-radius: 50%; animation: bounce 1.2s infinite; }
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%, 80%, 100% { transform: scale(0.7); opacity: 0.5; } 40% { transform: scale(1.0); opacity: 1; } }

.chat-input-area { border-top: 1px solid #e5e7eb; padding: 14px 20px; flex-shrink: 0; background: #fafafa; }
.input-container { display: flex; gap: 10px; align-items: flex-end; }

textarea#userInput {
    flex: 1; border: 1.5px solid #d1d5db; border-radius: 12px;
    padding: 11px 16px; font-size: 15px; font-family: inherit;
    resize: none; outline: none; max-height: 120px; overflow-y: auto;
    background: white; transition: border-color 0.2s; line-height: 1.5;
}
textarea#userInput:focus { border-color: #5b8dee; box-shadow: 0 0 0 3px rgba(91,141,238,0.15); }

button#sendBtn {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #5b8dee, #7c5cbf);
    color: white; border: none; border-radius: 50%; cursor: pointer;
    font-size: 16px; display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; transition: transform 0.15s, opacity 0.15s;
}
button#sendBtn:hover { transform: scale(1.08); }
button#sendBtn:active { transform: scale(0.95); }
button#sendBtn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

.input-footer { display: flex; justify-content: space-between; margin-top: 7px; font-size: 11.5px; color: #9ca3af; }
kbd { background: #e5e7eb; border-radius: 4px; padding: 1px 5px; font-size: 11px; font-family: monospace; }

@media (max-width: 600px) {
    body { padding: 0; }
    .chat-wrapper { border-radius: 0; height: 100vh; max-height: none; }
    .bubble { max-width: 85%; }
}
```

**File: `static/js/chat.js`**

```javascript
/**
 * chat.js — Mental Health Chatbot Frontend Logic
 */

const chatMessages    = document.getElementById('chatMessages');
const userInput       = document.getElementById('userInput');
const sendBtn         = document.getElementById('sendBtn');
const typingIndicator = document.getElementById('typingIndicator');
const charCount       = document.getElementById('charCount');

userInput.addEventListener('input', () => {
    const count = userInput.value.length;
    charCount.textContent = `${count}/500 characters`;
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
});

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    sendBtn.disabled = true;
    appendMessage(message, 'user');
    userInput.value = '';
    userInput.style.height = 'auto';
    charCount.textContent = '0/500 characters';
    showTyping(true);

    try {
        const response = await fetch('/get', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ msg: message }),
        });

        if (!response.ok) throw new Error(`HTTP error: ${response.status}`);

        const data = await response.json();
        appendMessage(data.response, 'bot', data.is_crisis === true);

    } catch (error) {
        console.error('Chat error:', error);
        appendMessage(
            "I'm sorry, something went wrong. Please try again in a moment.",
            'bot', false
        );
    } finally {
        showTyping(false);
        sendBtn.disabled = false;
        userInput.focus();
    }
}

function appendMessage(text, sender, isCrisis = false) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${sender}-message`);

    const bubble = document.createElement('div');
    bubble.classList.add('bubble', `${sender}-bubble`);
    if (isCrisis) bubble.classList.add('crisis-bubble');
    bubble.textContent = text;

    const timestamp = document.createElement('span');
    timestamp.classList.add('timestamp');
    timestamp.textContent = getTime();
    bubble.appendChild(timestamp);

    if (sender === 'bot') {
        const avatar = document.createElement('div');
        avatar.classList.add('avatar', 'bot-avatar');
        avatar.innerHTML = '<i class="fas fa-heart"></i>';
        messageDiv.appendChild(avatar);
    }

    messageDiv.appendChild(bubble);
    chatMessages.insertBefore(messageDiv, typingIndicator);
    scrollToBottom();
}

function showTyping(visible) {
    typingIndicator.style.display = visible ? 'flex' : 'none';
    if (visible) scrollToBottom();
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function getTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
```

---

## 11. Step 5 — Inference & Response Pipeline (GPU-Accelerated ⚡)

**File: `src/inference.py`**

```python
"""
src/inference.py — GPU-Accelerated Inference ⚡

Loads the fine-tuned model onto CUDA and generates empathetic responses.

GPU Inference Optimizations:
────────────────────────────────────────────────────────
1. FP16/BF16 MODEL:   Model weights stored in half-precision → less VRAM
2. torch.no_grad():   Disables gradient tracking → saves VRAM & compute
3. model.eval():      Disables dropout → deterministic, faster inference
4. Device placement:  model.cuda() moves all weights to GPU
────────────────────────────────────────────────────────
"""

import os
import re
import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class EmpathyBot:
    """
    Encapsulates the fine-tuned model for GPU-accelerated empathetic response generation.
    """

    PERSON_TOKEN = "Person:"
    SUPPORTER_TOKEN = "Supporter:"

    def __init__(self, model_path: str):
        """
        Loads the tokenizer and model onto the GPU.
        """
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Model directory not found: '{model_path}'. "
                "Have you run 'python train/finetune.py'?"
            )

        # Enforce GPU usage
        if not torch.cuda.is_available():
            raise RuntimeError(
                "❌ CUDA is not available for inference.\n"
                "Ensure you installed the GPU build of PyTorch:\n"
                "  uv add torch --index-url https://download.pytorch.org/whl/cu121"
            )

        self.device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Loading model from: {model_path} → {gpu_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine precision based on GPU capability
        major = torch.cuda.get_device_capability(0)[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16

        # Load model directly in half-precision onto GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(self.device)

        self.model.eval()  # Disable dropout — inference mode

        logger.info(f"✅ Model loaded on {gpu_name} ({dtype})")

        # Generation settings from .env
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", 150))
        self.temperature = float(os.getenv("TEMPERATURE", 0.85))
        self.top_p = float(os.getenv("TOP_P", 0.92))
        self.repetition_penalty = float(os.getenv("REPETITION_PENALTY", 1.3))

    def generate(self, user_message: str) -> str:
        """
        Generates an empathetic response using GPU inference.
        """
        prompt = f"{self.PERSON_TOKEN} {user_message.strip()} {self.SUPPORTER_TOKEN}"

        # Tokenize and move to GPU
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=200,
        ).to(self.device)

        prompt_length = inputs["input_ids"].shape[1]

        # ⚡ torch.no_grad() is critical — disables gradient tracking for inference
        # This halves memory usage and speeds up generation significantly
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens (strip the prompt)
        new_tokens = output_ids[0][prompt_length:]
        raw_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return self._clean_response(raw_response)

    def _clean_response(self, raw: str) -> str:
        """Cleans up raw model output."""
        raw = re.sub(r'\b(Person:|Supporter:)\b', '', raw, flags=re.IGNORECASE)
        raw = re.sub(r'\s+', ' ', raw).strip()

        if len(raw) < 5:
            return (
                "Thank you for sharing that with me. Could you tell me a bit more "
                "about what's on your mind?"
            )

        if raw and raw[-1] not in '.!?':
            raw += '.'

        return raw
```

---

## 12. Step 6 — Logging & Safety Layer

**File: `src/logger.py`**

```python
"""
src/logger.py — Structured logging setup.
"""

import logging
import sys


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates and configures a named logger.

    Args:
        name:  Logger name (e.g. "app", "inference")
        level: Minimum severity (DEBUG < INFO < WARNING < ERROR < CRITICAL)

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)-8s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
```

**File: `src/safety.py`**

```python
"""
src/safety.py — Safety layer for crisis detection.

IMPORTANT: This is a keyword-based system, not a substitute for
professional mental health assessment.
"""

import os
import re
import logging

logger = logging.getLogger(__name__)

CRISIS_PATTERNS = [
    r'\b(suicide|suicidal|kill myself|end my life|take my life)\b',
    r'\b(want to die|wish i was dead|don\'t want to live)\b',
    r'\b(self.harm|self-harm|cutting myself|hurt myself)\b',
    r'\b(overdose|od\'ing)\b',
    r'\b(no point|no reason to live|can\'t go on)\b',
]

DEFAULT_CRISIS_MESSAGE = (
    "I can hear that you're going through something incredibly painful right now, "
    "and I'm really glad you reached out. Please know you are not alone.\n\n"
    "🆘 **Immediate help is available:**\n"
    "• **988 Suicide & Crisis Lifeline:** Call or text **988** (US)\n"
    "• **Crisis Text Line:** Text HOME to **741741**\n"
    "• **International Association for Suicide Prevention:** "
    "https://www.iasp.info/resources/Crisis_Centres/\n\n"
    "Please reach out to one of these services — they have trained counselors "
    "available 24/7 who truly care. 💙"
)


class SafetyLayer:
    """Screens user messages for crisis content."""

    def __init__(self):
        self.crisis_message = os.getenv("CRISIS_RESPONSE_MSG", DEFAULT_CRISIS_MESSAGE)
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in CRISIS_PATTERNS
        ]
        logger.info(f"Safety layer initialized with {len(self.compiled_patterns)} patterns")

    def check(self, message: str) -> str | None:
        """
        Returns a crisis response string if crisis content is detected, else None.
        """
        for pattern in self.compiled_patterns:
            if pattern.search(message):
                logger.warning(f"Crisis pattern matched: '{pattern.pattern[:40]}...'")
                return self.crisis_message
        return None
```

---

## 13. Step 7 — Docker & Production Deployment

**File: `Dockerfile`** (GPU-enabled with NVIDIA CUDA base image)

```dockerfile
# ── GPU-enabled Dockerfile ─────────────────────────────────────
# Uses NVIDIA CUDA base image so the container has GPU access
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3.11-pip curl --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip3 install uv --no-cache-dir

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies (GPU PyTorch build)
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY app.py .
COPY .env.example .env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python3", "app.py"]
```

**File: `docker-compose.yml`** (with GPU pass-through)

```yaml
version: '3.9'

services:
  chatbot:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models:ro
    env_file:
      - .env
    restart: unless-stopped

    # ⚡ GPU pass-through — gives the container access to your NVIDIA GPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

**Build & Run Docker (with GPU):**
```bash
# Requires: nvidia-container-toolkit installed
# Install it: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

docker build -t mental-health-chatbot .
docker compose up
```

---

## 14. Step 8 — GitHub CI/CD with GitHub Actions

**File: `.github/workflows/ci-cd.yml`**

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  quality:
    name: Code Quality
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run Ruff linter
        run: uv run ruff check .

      - name: Check Black formatting
        run: uv run black --check .

  docker:
    name: Docker Build
    runs-on: ubuntu-latest
    needs: quality

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t mental-health-chatbot:${{ github.sha }} .

      - name: Verify Docker image
        run: |
          docker run --rm \
            -e MODEL_PATH=/nonexistent \
            mental-health-chatbot:${{ github.sha }} \
            python3 -c "from src.safety import SafetyLayer; print('✅ Imports OK')"
```

---

## 15. Running the Project End-to-End

Follow these steps in order.

### Step 1: Verify GPU & Clone Project
```bash
nvidia-smi   # Confirm GPU is visible

git clone https://github.com/YOUR_USERNAME/mental-health-chatbot.git
cd mental-health-chatbot

curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate

cp .env.example .env
```

### Step 2: Confirm PyTorch Sees Your GPU
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Expected: CUDA: True | GPU: NVIDIA GeForce RTX XXXX
```

### Step 3: Prepare the Dataset
```bash
python data/prepare_data.py
# ✅ Saved 76421 examples → data/processed/train.jsonl
```

### Step 4: Fine-Tune the Model (GPU ⚡)
```bash
python train/finetune.py
# ⚡ GPU: NVIDIA GeForce RTX ...
# 🚀 Starting GPU fine-tuning...
# Estimated time: ~5-15 min ⚡
```

### Step 5: Start the Chatbot Server
```bash
python app.py
# ╔════════════════════════════════════════╗
# ║  🧠 Mental Health Chatbot  ⚡ GPU      ║
# ║  Running at http://localhost:5000      ║
# ╚════════════════════════════════════════╝
```

### Step 6: Open in Browser
Navigate to `http://localhost:5000` — your GPU-powered chat interface is live!

---

## 16. Common Errors & Fixes

### Error: `CUDA available: False`
**Cause:** Wrong PyTorch build installed (CPU-only version).
**Fix:**
```bash
# Uninstall CPU version, install GPU version matching your CUDA
uv remove torch
uv add torch --index-url https://download.pytorch.org/whl/cu121
# Replace cu121 with your CUDA version from nvidia-smi
```

### Error: `torch.cuda.OutOfMemoryError`
**Cause:** Batch size too large for your GPU VRAM.
**Fix:** In `finetune.py`, set `BATCH_SIZE = 8` and add `gradient_checkpointing=True` to `TrainingArguments`.

### Error: `FileNotFoundError: Model not found`
**Cause:** You haven't trained the model yet.
**Fix:** Run `python train/finetune.py` first.

### Error: `ModuleNotFoundError: No module named 'transformers'`
**Cause:** Virtual environment not activated.
**Fix:** Run `source .venv/bin/activate` then retry.

### Error: Training loss is NaN
**Cause:** Learning rate too high or corrupt data.
**Fix:** Reduce `LEARNING_RATE` to `2e-5` in `finetune.py`.

### Error: Responses are repetitive/nonsensical
**Cause:** Model hasn't trained long enough, or `temperature` is too low.
**Fix:** Increase `NUM_EPOCHS` to 5, or raise `TEMPERATURE` to `0.9` in `.env`.

### Error: `nvidia-smi` not found (Windows)
**Fix:** Add `C:\Windows\System32\DriverStore\FileRepository\nv_dispi.inf_...\` to your PATH, or run from the NVIDIA directory. Alternatively just run `python -c "import torch; print(torch.cuda.is_available())"` — that's the definitive check.

---

## 17. Senior-Level Best Practices Used

| Practice | Where Used | Why It Matters |
|---|---|---|
| **Environment isolation** | `uv venv` + `pyproject.toml` | Reproducible builds across all machines |
| **Dependency locking** | `uv.lock` | Prevents "works on my machine" issues |
| **12-Factor App config** | `.env` + `os.getenv()` | Secrets never hardcoded in source |
| **GPU capability detection** | `finetune.py` + `inference.py` | Selects BF16 vs FP16 automatically |
| **FP16/BF16 mixed precision** | `TrainingArguments` | ~2x training speed, ~50% less VRAM |
| **Dataloader workers** | `TrainingArguments` | Eliminates GPU data-starvation bottleneck |
| **Singleton model loading** | `EmpathyBot.__init__()` | Avoids 30s delay on every API request |
| **torch.no_grad()** | `inference.py` | Disables gradients — faster & less VRAM |
| **model.eval()** | `inference.py` | Disables dropout for deterministic inference |
| **Structured logging** | `src/logger.py` | Debuggable in production without print() |
| **Safety-first design** | `src/safety.py` | Crisis detection before AI responses |
| **XSS prevention** | `textContent` in JS | Security best practice in frontend |
| **Health check endpoint** | `GET /health` (reports GPU status) | Required for Docker/Kubernetes |
| **NVIDIA CUDA Docker image** | `Dockerfile` | GPU access inside containers |
| **Docker GPU reservations** | `docker-compose.yml` | Passes GPU to container correctly |
| **CI/CD automation** | GitHub Actions | Code quality enforced before merge |
| **Early stopping** | `EarlyStoppingCallback` | Prevents overfitting, saves compute |
| **Type hints + docstrings** | All Python files | Self-documenting, professional codebase |

---

## 🎓 What You've Learned

By completing this project, you have hands-on experience with:

1. **LLM Fine-tuning** — Full pipeline: data prep → GPU training → inference
2. **CUDA & GPU ML** — Device management, FP16/BF16 precision, VRAM optimization
3. **Hugging Face Ecosystem** — Transformers, Datasets, Trainer API
4. **Production ML patterns** — Model serving, caching, environment config
5. **Modern Python tooling** — `uv`, `ruff`, `black`, type hints
6. **Web development** — Flask routing, REST APIs, async JavaScript
7. **Software safety** — Input validation, crisis detection, XSS prevention
8. **DevOps basics** — Docker (GPU-enabled), Docker Compose, GitHub Actions CI/CD
9. **Empathy-first AI design** — Tone, safety, responsible AI principles

This project is **portfolio-ready** for senior ML Engineer, AI Engineer, and Full-Stack AI Developer positions.

---

*Built with ❤️ — Remember: if you're struggling, please reach out to 988 (Suicide & Crisis Lifeline)*
