import torch
from torch.nn import Embedding
from torch.utils.data import DataLoader

from llm import GPTDataset, get_tokenizer
from llm.models.selfattention import SelfAttention
from config.paths import DATA_DIR

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
RAW_DIR = DATA_DIR / "raw"
RAW_FILE = RAW_DIR / "the-verdict.txt"

MAX_LENGTH = 4        # sequence length (tokens per sample)
CONTEXT_LENGTH = 4    # positional embedding context (usually = MAX_LENGTH)
BATCH_SIZE = 8        # number of sequences per batch
OUTPUT_DIM = 256      # embedding dimension
VOCAB_SIZE = 50257    # GPT-2 BPE vocab size (tiktoken gpt2)


def create_dataloader(
    text: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader over GPT-style sliding-window token sequences.

    Args:
        text: Raw input text.
        batch_size: Number of sequences per batch.
        max_length: Context window size (tokens per sequence).
        stride: Step size for sliding window over tokens.
        shuffle: Shuffle dataset between epochs (True for training).
        drop_last: Drop last incomplete batch if needed.
        num_workers: DataLoader workers.

    Returns:
        torch.utils.data.DataLoader
    """
    tokenizer = get_tokenizer()
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


def main() -> None:
    # -------------------------------------------------------------------
    # Load raw text
    # -------------------------------------------------------------------
    raw_text = RAW_FILE.read_text(encoding="utf-8")

    # -------------------------------------------------------------------
    # Create embedding layers
    # -------------------------------------------------------------------
    token_embedding_layer: Embedding = Embedding(VOCAB_SIZE, OUTPUT_DIM)
    pos_embedding_layer: Embedding = Embedding(CONTEXT_LENGTH, OUTPUT_DIM)

    # -------------------------------------------------------------------
    # Create DataLoader
    # -------------------------------------------------------------------
    dataloader = create_dataloader(
        raw_text,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        stride=MAX_LENGTH,
        shuffle=False,  # deterministic for this demo
    )

    # Get a single batch
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("Token IDs\n", inputs, "\n")
    print("Input Shape\n", inputs.shape, "\n")  # (batch_size, max_length)

    # -------------------------------------------------------------------
    # Embed tokens
    # -------------------------------------------------------------------
    token_embeddings: Embedding = token_embedding_layer(inputs)
    print("Token Embeddings Shape\n", token_embeddings.shape, "\n")
    # -> (batch_size, max_length, OUTPUT_DIM)

    # Positional embeddings (same for every sequence in batch)
    positions = torch.arange(CONTEXT_LENGTH, device=inputs.device)
    position_embeddings = pos_embedding_layer(positions)
    print("Position Embeddings Shape\n", position_embeddings.shape, "\n")
    # -> (max_length, OUTPUT_DIM)

    # Broadcast position embeddings over batch dimension
    input_embeddings = token_embeddings + position_embeddings
    print("Input Embeddings Shape\n", input_embeddings.shape, "\n")




if __name__ == "__main__":
    main()

