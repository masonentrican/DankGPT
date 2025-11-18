import math
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    Feed-forward layer.

    This layer is a feed-forward layer.
    It is used to introduce non-linearity into the model.
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.get("emb_dim"), 4 * cfg.get("emb_dim")),
            GELU(),
            nn.Linear(4 * cfg.get("emb_dim"), cfg.get("emb_dim")),
        )

    def forward(self, x):
        return self.layers(x)


class GELU(nn.Module):
    """
    GELU activation function.

    This activation function is a smooth approximation of the ReLU function.
    Useful for introducing non-linearity into the model like in the feed-forward layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))