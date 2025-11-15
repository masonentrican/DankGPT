import tiktoken
import GPTDataset
from GPTDataset import GPTDataset
from torch.utils.data import DataLoader

# Create a tokenized GPT dataset from text input
#
# BatchSize: Number of tokens per batch
# MaxLength: Context size
# Stride: Number of positions the input shifts across batches
# Shuffle: Shufles dataset - usually always false
# DropLast: Drops final batch if shorter than batch size
def createDataLoader(text, batchSize=4, maxLength=256, stride=128,
                     shuffle=True, dropLast=True, numWorkers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, maxLength, stride)
    dataloader = DataLoader(
        dataset,
        batchSize,
        shuffle=shuffle,
        drop_last=dropLast,
        num_workers=numWorkers
    )

    return dataloader


# Main
with open("the-verdict.txt", mode="r", encoding="utf-8") as f:
    rawText = f.read()

dataloader = createDataLoader(
    rawText,
    batchSize=8,
    maxLength=4,
    stride=4,
    shuffle=False
)

# Convert dataloader to py iterator to fetch the next entry
dataItr = iter(dataloader)
inputs, targets = next(dataItr)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
