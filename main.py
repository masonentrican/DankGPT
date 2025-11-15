import tiktoken
from torch.nn import Embedding
import GPTDataset
import torch
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

maxLength     = 4       # How many tokens per text sample
contextLength = 4       # For pos embedding context. In practice same as length
batchSize     = 8       # How many batches of tokens from text sample
outputDim     = 256     # How many dimensions (GPT3 used 12288 got dam)
vocabSize     = 50257   # Matches BPE tiktoken vocabsize

# Token Embedding Layer
# The embedding layer is a randomly seeded tensor (matrix effectively)
# of random weights. When embedding tokens to this layer, the token id value
# effectively maps to the key id in the embedding tensor, where the keys start
# at zero

"""
Embedding Lookup
-----------------------------

Weight matrix of the embedding layer (example):

    Token ID |    Embedding Vector
    -----------------------------------------------
         0   |  0.3374   -0.1778   -0.1690
         1   |  0.9178    1.5810    1.3010
         2   |  1.2753   -0.2010    0.1606
         3   | -0.4015    0.9666   -1.1481
         5   |  1.1589    0.3255   -0.6315
         7   | -2.8400   -0.7849   -1.4096

Token IDs to embed:

    ["fox", "jumps", "over", "fox"]
        ↓      ↓       ↓      ↓
      [ 2,     3,      5,     2 ]

Embedding lookup:

    ID 2 → [ 1.2753  -0.2010   0.1606 ]
    ID 3 → [ -0.4015  0.9666  -1.1481 ]
    ID 5 → [ 1.1589   0.3255  -0.6315 ]
    ID 2 → [ 1.2753  -0.2010   0.1606 ]   (same ID → same embedding)

Key idea:
    The embedding layer is just a lookup table.
    Passing a token ID returns the corresponding row in the weight matrix.
"""
token_embedding_layer: Embedding = torch.nn.Embedding(vocabSize,outputDim)


# Positional embedding layer. Same concept as above, but for relative positions
# of tokens within a sequence for more context
pos_embedding_layer: Embedding = torch.nn.Embedding(contextLength,outputDim)



# Dataloader object
dataloader = createDataLoader(
    rawText,
    batchSize=batchSize,
    maxLength=maxLength,
    stride=maxLength,
    shuffle=False
)

# Convert dataloader to py iterator to fetch the entries
dataItr = iter(dataloader)
inputs, targets = next(dataItr)
print("Token IDs\n", inputs, "\n")
print("Input Shape\n", inputs.shape,"\n") # Shape shows dimensions of tensor ( batchsize x maxLength)

# Embed the tokens
tokenEmbeddings = token_embedding_layer(inputs)
print("Token Embeddings Shape\n",tokenEmbeddings.shape,"\n")

# Create pos embedding layer
positionEmbeddings = pos_embedding_layer(torch.arange(contextLength))
print("Position Embeddings Shape\n",positionEmbeddings.shape,"\n")

# Add the position and token embeddings together to get our final final input embeddings
inputEmbeddings = tokenEmbeddings + positionEmbeddings
print("Input Embeddings Shape\n", inputEmbeddings.shape,"\n")

print("\nInputs have been embedded! Ready for LLM munching.")
