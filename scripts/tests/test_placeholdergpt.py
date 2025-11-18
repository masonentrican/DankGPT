import torch
import tiktoken
from llm.config import load_config
from llm.models.placeholdergptmodel import PlaceholderGPTModel


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
model = PlaceholderGPTModel(load_config("gpt_config_124m"))
logits = model(batch)

# Debug prints
print("-----------------------Placeholder GPT Model Test-----------------------")
print("Batch:\n ", batch)
print("Logits:\n ", logits)
print("Logits Shape:\n ", logits.shape)