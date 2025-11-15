import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader 

class GPTDataset(Dataset):
    """
    Creates a token sequence by sliding window.
    Handles tokenization with tiktoken Byte Pair Encoding
    Handles windowing of chunk sequences.
    """

    def __init__(self,
                 text: str,
                 tokenizer: tiktoken.Encoding,
                 max_length: int,
                 stride: int):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length # Non-overlapping chunks by default

        # Tokenize text content
        self.tokens = tokenizer.encode(text)
        self.num_tokens = len(self.tokens)

        # Pre compute how many windows fit
        self.num_windows = (self.num_tokens - max_length) // self.stride

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.max_length

        x = self.tokens[start:end]
        y = self.tokens[start+1:end+1]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        ) 
