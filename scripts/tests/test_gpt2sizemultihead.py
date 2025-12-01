import torch
from llm.models.selfattention import MultiHeadAttention
from llm.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

torch.manual_seed(123)

# GPT-2 style configuration
d_in = 768
d_out = 768
context_length = 1024
num_tokens = 1024  # Number of tokens in sequence (can be up to context_length)

# Generate random embeddings with correct dimensions: [num_tokens, d_in]
input_embeddings = torch.randn(num_tokens, d_in)

batch = torch.stack((input_embeddings, input_embeddings), dim=0)

multihead_attn = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=12)
context_vectors = multihead_attn(batch)

logger.debug(f"Context Vectors:\n {context_vectors}")
logger.debug(f"Context Vectors Shape:\n {context_vectors.shape}")