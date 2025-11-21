"""
Unified configuration file for GPT-2 model variants.

All model configurations are available directly:
    from llm.config import GPT2_SMALL, GPT2_MEDIUM, GPT2_LARGE, GPT2_XLARGE
"""

# GPT-2 Small (124M parameters)
SMOOTHBRAIN = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 256,   # Context length
    "emb_dim": 768,           # Embedding dimension
    "num_heads": 12,          # Number of attention heads
    "num_layers": 12,         # Number of layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # Query-Key-Value bias
}

# GPT-2 Small (124M parameters)
GPT2_SMALL = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 1024,   # Context length
    "emb_dim": 768,           # Embedding dimension
    "num_heads": 12,          # Number of attention heads
    "num_layers": 12,         # Number of layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # Query-Key-Value bias
}

# GPT-2 Medium (350M parameters)
GPT2_MEDIUM = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 1024,   # Context length
    "emb_dim": 1024,          # Embedding dimension
    "num_heads": 16,          # Number of attention heads
    "num_layers": 24,         # Number of layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # Query-Key-Value bias
}

# GPT-2 Large (774M parameters)
GPT2_LARGE = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 1024,   # Context length
    "emb_dim": 1280,          # Embedding dimension
    "num_heads": 20,          # Number of attention heads
    "num_layers": 36,         # Number of layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # Query-Key-Value bias
}

# GPT-2 Extra Large (1.5B parameters)
GPT2_XLARGE = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 1024,   # Context length
    "emb_dim": 1600,          # Embedding dimension
    "num_heads": 25,          # Number of attention heads
    "num_layers": 48,         # Number of layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # Query-Key-Value bias
}

__all__ = [
    "SMOOTHBRAIN",
    "GPT2_SMALL",
    "GPT2_MEDIUM",
    "GPT2_LARGE",
    "GPT2_XLARGE",
]
