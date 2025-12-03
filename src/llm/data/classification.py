import torch
from torch.utils.data import Dataset, DataLoader 
import pandas as pd
from llm.utils.tokenizer import get_tokenizer

class ClassificationDataset(Dataset):
    """
    Creates a dataset for classification tasks.

    Loads data from csv, tokenizes the text, and pads or truncates the sequences
    to the longest length in the dataset.

    Args:
        csv_file: Path to the CSV file containing the text and labels.
        tokenizer: Tokenizer to use for encoding the text.
        max_length: Maximum length of the text. If None, the longest text in the dataset will be used.
        pad_token_id: Token ID to use for padding the text.
    """
    def __init__(self, csv_file, tokenizer, max_length = None, pad_token_id = 50256):
        
        self.data = pd.read_csv(csv_file)

        # Pretokenize the text
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        # Truncate the text if longer than max_length
        if max_length is None:
            self.max_length = self._longest_text_length()
        else:
            self.max_length = max_length

            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest length in the dataset
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        label = self.data.iloc[idx]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def _longest_text_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length