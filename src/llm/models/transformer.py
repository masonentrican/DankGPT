import torch
import torch.nn as nn

from llm.models.selfattention import MultiHeadAttention
from llm.models.feedforward import FeedForward
from llm.models.normalization import Normalization

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            dim_in=cfg.get("emb_dim"),
            dim_out=cfg.get("emb_dim"),
            context_length=cfg.get("context_length"),
            dropout=cfg.get("drop_rate"),
            num_heads=cfg.get("num_heads"),
            qkv_bias=cfg.get("qkv_bias"),
        )

        self.feedforward = FeedForward(cfg)
        self.normalization1 = Normalization(cfg.get("emb_dim"))
        self.normalization2 = Normalization(cfg.get("emb_dim"))
        self.drop_shortcut = nn.Dropout(cfg.get("drop_rate"))


    def forward(self, x):
        """
        Forward pass for the transformer block.

        Computes the attention and feedforward outputs, applies layer normalization
        before each of these componenets to regularize the model and prevent overfitting.
        Then applies dropout to the shortcut connections to assist in gradient flow.
        
        Args:
            x: Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        
        # Shortcut connection for the attention block
        shortcut = x
        x = self.normalization1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Shortcut connection for the feedforward block
        shortcut = x
        x = self.normalization2(x)
        x = self.feedforward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x