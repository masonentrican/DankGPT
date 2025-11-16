import torch
from llm.models.selfattention import SelfAttention

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

print("d_in: ",d_in)
print("d_out: ",d_out)

self_attn = SelfAttention(d_in,d_out)

print(self_attn(input_embeddings))

