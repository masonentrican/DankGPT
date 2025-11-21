import torch

from llm.config import GPT2_SMALL
from llm.models.transformer import Transformer

torch.manual_seed(123)
x = torch.randn(2, 4, 768)
block = Transformer(GPT2_SMALL)
output = block(x)

print("Input Shape: ", x.shape)
print("Output Shape: ", output.shape)