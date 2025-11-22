"""
DataLoader utilities for GPT-style models.
"""

import tiktoken
from torch.utils.data import DataLoader

from llm.data.dataset import GPTDataset


def create_dataloader(
    text: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader over GPT-style sliding-window token sequences.

    Args:
        text: Raw input text.
        batch_size: Number of sequences per batch.
        max_length: Context window size (tokens per sequence).
        stride: Step size for sliding window over tokens.
        shuffle: Shuffle dataset between epochs (True for training).
        drop_last: Drop last incomplete batch if needed.
        num_workers: DataLoader workers.

    Returns:
        torch.utils.data.DataLoader
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader

