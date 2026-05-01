#!/usr/bin/env python3
"""
Quick start script for Mental Health Chatbot
Runs the Flask app and waits for model training if needed
"""

import os
import sys
import subprocess
from pathlib import Path

def check_model_exists():
    """Check if fine-tuned model exists"""
    model_path = Path("models/empathy-model")
    return (model_path / "pytorch_model.bin").exists()

def check_data_exists():
    """Check if training data is prepared"""
    data_path = Path("data/processed")
    return (data_path / "train.jsonl").exists()

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║     🧠 Mental Health Support Chatbot - Quick Start             ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check data
    if not check_data_exists():
        print("📥 Preparing dataset...")
        result = subprocess.run([sys.executable, "data/prepare_data.py"])
        if result.returncode != 0:
            print("❌ Failed to prepare data")
            return 1
    
    # Step 2: Check model
    if not check_model_exists():
        print("""
⚠️  Model not found! Starting fine-tuning...
This may take 30-60 minutes on CPU.
For faster training, use GPU or Google Colab.
        """)
        result = subprocess.run([sys.executable, "train/finetune.py"])
        if result.returncode != 0:
            print("❌ Model training failed")
            return 1
    else:
        print("✅ Model found")
    
    # Step 3: Run the app
    print("""
🚀 Starting Flask app...
📌 Open http://localhost:5000 in your browser
    """)
    subprocess.run([sys.executable, "app.py"])
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
