import torch
import torch.nn as nn

class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, emb_dim: int, drop_rate: float):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        # in_idx: [B, T]
        batch_size, seq_len = in_idx.shape
        tok = self.tok_emb(in_idx)  # [B, T, D]
        pos = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # [T, D]
        x = tok + pos
        return self.drop(x)