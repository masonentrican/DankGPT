import torch
from llm.models.causalattention import CausalAttention

torch.manual_seed(123)

# TEMP: Hardcoded embeddings (shape [6, 3])
input_embeddings = torch.tensor([
    [0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55],
])

d_in = input_embeddings.shape[1]
d_out = 2

batch = torch.stack((input_embeddings, input_embeddings), dim=0)
context_length = batch.shape[1]

causal_attn = CausalAttention(d_in,d_out, context_length, 0.0)
context_vectors = causal_attn(batch)