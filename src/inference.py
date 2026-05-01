import os
import re
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class EmpathyBot:
    """
    Encapsulates the fine-tuned model for empathetic response generation.
    
    Design Pattern: We load the model ONCE in __init__ and reuse it.
    Loading a neural network from disk into RAM takes seconds. We don't
    want that delay on every user message.
    """

    REQUIRED_MODEL_FILES = ["config.json", "pytorch_model.bin"]
    REQUIRED_TOKENIZER_FILES = [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]

    @classmethod
    def validate_model_path(cls, model_path: str) -> None:
        """Ensure the model directory exists and contains saved weights/tokenizer files."""
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"Model directory not found: '{model_path}'. "
                "Have you run 'python train/finetune.py'?"
            )

        for required_file in cls.REQUIRED_MODEL_FILES:
            if not (path / required_file).is_file():
                raise FileNotFoundError(
                    f"Model directory is incomplete: '{model_path}'. "
                    f"Missing required file: '{required_file}'."
                )

        if not any((path / file_name).is_file() for file_name in cls.REQUIRED_TOKENIZER_FILES):
            raise FileNotFoundError(
                f"Model directory is missing tokenizer files: '{model_path}'. "
                "Please ensure you save the tokenizer with the model."
            )


    # These tokens mark the structure of our training data
    PERSON_TOKEN = "Person:"
    SUPPORTER_TOKEN = "Supporter:"

    def __init__(self, model_path: str):
        """
        Loads the tokenizer and model from disk.
        
        Args:
            model_path: Path to the fine-tuned model directory.
                        Must contain: config.json, pytorch_model.bin, tokenizer files
        
        Raises:
            FileNotFoundError: If model_path doesn't exist or is incomplete
        """
        self.validate_model_path(model_path)

        logger.info(f"Loading model from: {model_path}")
        
        # Detect available hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference device: {self.device}")

        # Load tokenizer (converts text ↔ token IDs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False  # Avoid sentencepiece/tiktoken requirement
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model and move to the appropriate device (GPU or CPU)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # torch_dtype reduces memory usage on GPU
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()  # Disable dropout layers — we're doing inference, not training

        # Load generation settings from environment (with sensible defaults)
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", 150))
        self.temperature = float(os.getenv("TEMPERATURE", 0.85))
        self.top_p = float(os.getenv("TOP_P", 0.92))
        self.repetition_penalty = float(os.getenv("REPETITION_PENALTY", 1.3))

        logger.info("✅ Model loaded and ready for inference")

    def generate(self, user_message: str) -> str:
        """
        Generates an empathetic response to the user's message.
        
        Args:
            user_message: The user's raw input text
        
        Returns:
            A clean, empathetic response string
        """
        # 1. Build the prompt using the same format as training data
        #    "Person: [user message] Supporter:"
        #    The model has learned to continue after "Supporter:"
        prompt = f"{self.PERSON_TOKEN} {user_message.strip()} {self.SUPPORTER_TOKEN}"

        # 2. Tokenize the prompt
        #    return_tensors="pt" → returns PyTorch tensors
        #    .to(self.device) → move tensors to GPU/CPU
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=200,  # Don't let the prompt itself be too long
        ).to(self.device)

        prompt_length = inputs["input_ids"].shape[1]  # Number of tokens in prompt

        # 3. Generate response
        #    torch.no_grad() disables gradient computation — saves memory
        #    We don't need gradients during inference (only during training)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                
                # do_sample=True enables stochastic (random) sampling
                # do_sample=False → greedy decoding (always picks highest prob word)
                # We use sampling for natural, varied responses
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                
                # Stop generating when we hit the end-of-text token
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 4. Decode only the NEW tokens (strip the prompt from output)
        new_tokens = output_ids[0][prompt_length:]
        raw_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # 5. Post-process to get a clean response
        return self._clean_response(raw_response)

    def _clean_response(self, raw: str) -> str:
        """
        Cleans up the raw model output.
        
        Models sometimes generate artifacts: repeated tokens, weird punctuation,
        leftover format strings. We clean those up here.
        
        Args:
            raw: Raw decoded string from the model
        
        Returns:
            Clean, presentable response
        """
        # Remove any leftover "Person:" or "Supporter:" tokens
        raw = re.sub(r'\b(Person:|Supporter:)\b', '', raw, flags=re.IGNORECASE)
        
        # Collapse multiple spaces/newlines
        raw = re.sub(r'\s+', ' ', raw).strip()

        # If the model generated nothing useful, return a default
        if len(raw) < 5:
            return (
                "Thank you for sharing that with me. Could you tell me a bit more "
                "about what's on your mind?"
            )

        # Ensure response ends with proper punctuation
        if raw and raw[-1] not in '.!?':
            raw += '.'

        return raw