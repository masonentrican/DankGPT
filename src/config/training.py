"""
Training configuration for GPT models.

Provides structured configuration for training parameters, making it easy to
experiment with different training setups and prepare for fine-tuning tasks.
"""

from typing import TypedDict, Optional


class TrainingConfig(TypedDict, total=False):
    """
    Training configuration dictionary.
    
    All fields are optional to allow partial configs, but typical usage
    should include all fields for a complete training setup.
    """
    # Data parameters
    batch_size: int              # Number of sequences per batch
    max_length: int              # Context window size (tokens per sequence)
    stride: int                  # Step size for sliding window over tokens
    train_ratio: float           # Fraction of data to use for training (0.0-1.0)
    
    # Training loop parameters
    num_epochs: int               # Number of training epochs
    learning_rate: float          # Learning rate for optimizer
    weight_decay: float           # Weight decay for optimizer
    eval_freq: int                # Frequency of evaluation (every N steps)
    eval_iter: int                # Number of batches to evaluate on
    
    # Generation parameters
    start_context: str            # Starting context for text generation samples
    
    # DataLoader parameters
    shuffle: bool                 # Shuffle dataset between epochs
    drop_last: bool               # Drop last incomplete batch
    num_workers: int              # DataLoader workers


# Default training configurations for common scenarios

# Quick test/development configuration
QUICK: TrainingConfig = {
    "batch_size": 8,
    "max_length": 4,
    "stride": 4,
    "train_ratio": 0.90,
    "num_epochs": 1,
    "learning_rate": 0.0004,
    "weight_decay": 0.1,
    "eval_freq": 5,
    "eval_iter": 5,
    "start_context": "Every effort moves you",
    "shuffle": True,
    "drop_last": True,
    "num_workers": 0,
}

# Standard training configuration
STANDARD: TrainingConfig = {
    "batch_size": 32,
    "max_length": 256,
    "stride": 128,
    "train_ratio": 0.90,
    "num_epochs": 10,
    "learning_rate": 0.0004,
    "weight_decay": 0.1,
    "eval_freq": 100,
    "eval_iter": 50,
    "start_context": "Every effort moves you",
    "shuffle": True,
    "drop_last": True,
    "num_workers": 0,
}

# Long training run configuration
LONG: TrainingConfig = {
    "batch_size": 64,
    "max_length": 512,
    "stride": 256,
    "train_ratio": 0.90,
    "num_epochs": 100,
    "learning_rate": 0.0003,
    "weight_decay": 0.1,
    "eval_freq": 500,
    "eval_iter": 100,
    "start_context": "Every effort moves you",
    "shuffle": True,
    "drop_last": True,
    "num_workers": 4,
}

