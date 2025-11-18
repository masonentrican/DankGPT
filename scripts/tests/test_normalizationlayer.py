import torch
import tiktoken
from llm.config import load_config
from llm.models.placeholdergptmodel import LayerNormalization


# Prepare batch (two training examples with 5 dimensions)
tokenizer = tiktoken.get_encoding("gpt2")
emb_dim = 5
batch = torch.randn(2, emb_dim)

# Test layer normalization with static seed
torch.manual_seed(123)
ln = LayerNormalization(emb_dim)
output = ln(batch)
mean = output.mean(dim=-1, keepdim=True)
variance = output.var(dim=-1, keepdim=True, unbiased=False)

# Debug prints without scientific notation to filter precision errors
# Expecting a mean of 0 and a variance of 1 to confirm the layer normalization is working
torch.set_printoptions(sci_mode=False)
print("-----------------------Layer Normalization Test-----------------------")
print("Mean:\n ", mean)
print("Variance:\n ", variance)