"""
Tokenizer utilities for GPT models.

Provides centralized tokenizer management to avoid duplication
and enable easy configuration changes.
"""

import tiktoken
from typing import Optional


# Global tokenizer instance (lazy-loaded singleton)
_tokenizer: Optional[tiktoken.Encoding] = None


def get_tokenizer(encoding_name: str = "gpt2") -> tiktoken.Encoding:
    """
    Get or create a tokenizer instance.
    
    Uses a singleton pattern to avoid creating multiple tokenizer instances.
    The tokenizer is cached after first creation.
    
    Args:
        encoding_name: Name of the tiktoken encoding to use. Defaults to "gpt2".
    
    Returns:
        tiktoken.Encoding: The tokenizer instance.
    
    Example:
        >>> tokenizer = get_tokenizer()
        >>> token_ids = tokenizer.encode("Hello, world!")
    """
    global _tokenizer
    
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding(encoding_name)
    
    return _tokenizer


def reset_tokenizer() -> None:
    """
    Reset the cached tokenizer instance if tokenizer type needs changing.
    """
    global _tokenizer
    _tokenizer = None

