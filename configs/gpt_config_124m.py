"""
GPT-2 124M parameter model configuration.

This configuration matches the GPT-2 Small model architecture.
"""

GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 1024,   # Context length
    "emb_dim": 768,           # Embedding dimension
    "num_heads": 12,            # Number of attention heads
    "num_layers": 12,           # Number of layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # Query-Key-Value bias
}

