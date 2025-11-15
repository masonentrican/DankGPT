import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

#
# GPTDataset uses the tiktoken Byte Pair Encoder to tokenize
# an input payload. Tokens are iterated over with a sliding
# window approach to store an input and target id list.
# Ultimately used to predict the next word from input data
#

class GPTDataset(Dataset):
    def __init__(self, text: str, tokenizer: tiktoken.Encoding,
                 maxLength: int, stride: int):
        self.inputIds = []
        self.targetIds = []

        tokenIds = tokenizer.encode(text)

        for i in range(0, len(tokenIds) - maxLength, stride):
            # Using sliding window, store pointers for current input
            # and target chunk. Target being 1 token after input
            inputChunk = tokenIds[i:i + maxLength]
            targetChunk = tokenIds[i + 1: i + maxLength + 1]

            # Store input and targets
            self.inputIds.append(torch.tensor(inputChunk))
            self.targetIds.append(torch.tensor(targetChunk))

    def __len__(self):
        return len(self.inputIds)

    def __getitem__(self, index):
        return self.inputIds[index], self.targetIds[index]
