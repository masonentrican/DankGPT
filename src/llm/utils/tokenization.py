"""
Tokenization utilities for text encoding and decoding.
"""

import torch


def text_to_token_ids(text, tokenizer):
    """
    Convert text to token IDs tensor with batch dimension.

    Args:
        text: Input text string.
        tokenizer: Tokenizer object with encode method.

    Returns:
        torch.Tensor: Token IDs with shape (1, seq_len).
    """
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Unsqueeze to add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs tensor to text string.

    Args:
        token_ids: Token IDs tensor with shape (batch_size, seq_len) or (1, seq_len).
        tokenizer: Tokenizer object with decode method.

    Returns:
        str: Decoded text string.
    """
    flat = token_ids.squeeze(0)  # Squeeze to remove batch dimension
    return tokenizer.decode(flat.tolist())

