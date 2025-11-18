import math
import torch
import torch.nn as nn

from llm.config import load_config
from llm.models.embeddings import GPTEmbedding
from llm.models.normalization import LayerNormalization

class PlaceholderGPTModel(nn.Module):
    """
    Placeholder GPT model.

    This model is a placeholder for the actual GPT model.
    It is used to test the training and inference pipeline.

    Args:
        cfg: Configuration dictionary.

    Returns:
        torch.Tensor: Output logits.
    """
    def __init__(self, cfg):
        super().__init__()
        self.emb = GPTEmbedding(
            vocab_size=cfg.get("vocab_size"),
            context_length=cfg.get("context_length"),
            emb_dim=cfg.get("emb_dim"),
            drop_rate=cfg.get("drop_rate")
        )

        # Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[PlaceholderTransformerBlock(cfg) # TODO: Replace with actual transformer block
                for _ in range(cfg.get("n_layers"))]
        )

        # Final layer norm and output head
        self.final_norm = LayerNormalization(cfg.get("emb_dim"))
        self.out_head = nn.Linear(cfg.get("emb_dim"), cfg.get("vocab_size"), bias=False)

    def forward(self, in_idx):
        """
        Forward pass.

        Describes the data flow through the model. Computes the token embeddings, positional
        embeddings, and applies dropout. Then passes the output through the transformer blocks
        and final layer norm. Finally, produces logits with the linear output layer.

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
    

class PlaceholderTransformerBlock(nn.Module):
    """
    Placeholder transformer block.

    This block is a placeholder for the actual transformer block.
    It is used to test the training and inference pipeline.

    Args:
        cfg: Configuration dictionary.

    Returns:
        torch.Tensor: Output tensor.
    """
    def __init__(self, cfg): # TODO: Replace with actual transformer block
        super().__init__()

    def forward(self, x): # TODO: Replace with actual transformer block forward pass
        return x