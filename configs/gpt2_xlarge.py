"""
GPT-2 124M parameter model configuration.

This configuration matches the GPT-2 Extra Large model architecture.
"""

GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 1024,   # Context length
    "emb_dim": 1600,          # Embedding dimension
    "num_heads": 25,          # Number of attention heads
    "num_layers": 48,         # Number of layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # Query-Key-Value bias
}

