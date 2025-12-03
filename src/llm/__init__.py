"""
DankGPT - A GPT-style language model implementation.

This package provides:
- Model architectures (GPTModel)
- Data handling (datasets, loaders)
- Training utilities
- Text generation
- Utility functions (tokenization, device management, checkpoints)
"""

# Model exports
from llm.models import GPTModel

# Data exports
from llm.data import GPTDataset, create_dataloader, download_text

# Training exports
from llm.training import (
    calc_language_loss_batch,
    calc_language_loss_loader,
    evaluate_language_model,
    train_language_model_simple,
)

# Generation exports
from llm.generation import generate_text, generate_and_print_sample

# Utility exports
from llm.utils import (
    get_device,
    get_tokenizer,
    load_checkpoint,
    save_checkpoint
)

__all__ = [
    # Models
    "GPTModel",
    # Data
    "GPTDataset",
    "create_dataloader",
    "download_text",
    # Training
    "calc_language_loss_batch",
    "calc_language_loss_loader",
    "evaluate_language_model",
    "train_language_model_simple",
    # Generation
    "generate_text",
    "generate_and_print_sample",
    # Utilities
    "get_device",
    "get_tokenizer",
    "load_checkpoint",
    "save_checkpoint",
]
