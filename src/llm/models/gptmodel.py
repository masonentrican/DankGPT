import math
import torch
import torch.nn as nn

from llm.models.embeddings import GPTEmbedding
from llm.models.normalization import Normalization
from llm.models.transformer import Transformer

class GPTModel(nn.Module):
    """
    GPT-2 style language model.

    The model consists of an embedding layer, dropout, a series of transformer blocks, and a final layer norm and output head.
    The embedding layer is used to embed the input tokens into a vector space.
    The transformer blocks are used to process the input tokens.
    The final layer norm and output head are used to produce the output logits.

    Args:
        cfg: Configuration dictionary.

    Returns:
        torch.Tensor: Output logits.
    """
    def __init__(self, cfg):
        """
        Initializes the GPT-2 style language model, but does not yet compute
        """   
        super().__init__()

        # Embedding layer with optional dropout
        self.emb = GPTEmbedding(
            vocab_size=cfg.get("vocab_size"),
            context_length=cfg.get("context_length"),
            emb_dim=cfg.get("emb_dim"),
            drop_rate=cfg.get("drop_rate")
        )

        # Series of Transformer blocks driven by the configuration's number of layers
        self.trf_blocks = nn.Sequential(
            *[Transformer(cfg)
                for _ in range(cfg.get("num_layers"))]
        )

        # Final layer normalization and output head
        self.final_norm = Normalization(cfg.get("emb_dim"))
        self.out_head = nn.Linear(cfg.get("emb_dim"), cfg.get("vocab_size"), bias=False)

    def forward(self, in_idx):
        """
        Forward pass.

        Computes the token embeddings, positional embeddings, and applies dropout.
        Then passes the output through the transformer blocks and final layer norm.
        Returns logits with the linear output layer.

        Args:
            in_idx: Input token indices.

        Returns:
            torch.Tensor: Output logits.
        """
        x = self.emb(in_idx)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits