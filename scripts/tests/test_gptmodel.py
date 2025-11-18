import torch
import tiktoken
from llm.config import load_config
from llm.models.gptmodel import GPTModel


# Prepare batch
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

# Test model with static seed
torch.manual_seed(123)
model = GPTModel(load_config("gpt2_xlarge"))
logits = model(batch)

# Analaytics
total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = (
    total_params - sum(p.numel() for p in model.out_head.parameters())
)

total_size_bytes = total_params * 4 # Assumes float32, 4 bytes per parameter
total_size_mb = total_size_bytes / (1024 * 1024)


# Debug prints
print("-----------------------GPT Model Test-----------------------")
print("Batch:\n ", batch)
print("Logits Shape:\n ", logits.shape)
print("Logits:\n ", logits)
print("Total parameters:\n ", total_params)
print("Total trainable parameters:\n ", total_trainable_params)
print("Token embedding layer shape:\n ", model.emb.tok_emb.weight.shape)
print("Output layer shape:\n ", model.out_head.weight.shape)
print("Total size in MB:\n ", total_size_mb)

"""
EXPECTED OUTPUT from 124m model:


Batch:
  tensor([[6109, 3626, 6100,  345],             # The token id's of Text 1 and 2
        [6109, 1110, 6622,  257]])
Logits Shape:
  torch.Size([2, 4, 50257])                     # [B, T, V] - Batch size, Token length, Vocabulary size
Logits:
  tensor([[[ 0.3613,  0.4222, -0.0711,  ...,  0.3483,  0.4661, -0.2838],
         [-0.1792, -0.5660, -0.9485,  ...,  0.0477,  0.5181, -0.3168],
         [ 0.7120,  0.0332,  0.1085,  ...,  0.1018, -0.4327, -0.2553],
         [-1.0076,  0.3418, -0.1190,  ...,  0.7195,  0.4023,  0.0532]],

        [[-0.2564,  0.0900,  0.0335,  ...,  0.2659,  0.4454, -0.6806],
         [ 0.1230,  0.3653, -0.2074,  ...,  0.7705,  0.2710,  0.2246],
         [ 1.0558,  1.0318, -0.2800,  ...,  0.6936,  0.3205, -0.3178],
         [-0.1565,  0.3926,  0.3288,  ...,  1.2630, -0.1858,  0.0388]]],
       grad_fn=<UnsafeViewBackward0>)
Total parameters:
  163009536
Total trainable parameters:
  124412160
Token embedding layer shape:
  torch.Size([50257, 768])
Output layer shape:
  torch.Size([50257, 768])
Total size in MB:
  621.83203125

NOTE: Total parameters exceed the configured model parameter size (124M) because weight tying
      has not been implemented. Weight tying shares weights between the input embedding layer
      and the output projection layer, reducing the total parameter count. While this technique
      reduces memory and computational complexity, it may reduce model/training performance.

      Parameter breakdown (without weight tying):
      
      Embeddings: (vocab_size x emb_dim) + (context_length x emb_dim) = (50,257 x 768) + (1,024 x 768) = 39.4M

      Transformer blocks (12x): Each block has self-attention (QKV: 768→2304, out: 768→768) and 
      MLP (768→3072→768) plus 2 layer norms = ~7.1M per block → 84.1M total

      Final layer norm: 768 + 768 = 1.5K
      Output head: vocab_size x emb_dim = 50,257 x 768 = 38.6M
      
      Total without weight tying: 39.4M + 84.1M + 0.0015M + 38.6M ≈ 163M
      Total With weight tying:            39.4M + 84.1M + 0.0015M ≈ 124.4M
"""