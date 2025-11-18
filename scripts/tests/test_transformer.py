import torch

from llm.config import load_config
from llm.models.transformer import Transformer

torch.manual_seed(123)
x = torch.randn(2, 4, 768)
block = Transformer(load_config("gpt_config_124m"))
output = block(x)

print("Input Shape: ", x.shape)
print("Output Shape: ", output.shape)