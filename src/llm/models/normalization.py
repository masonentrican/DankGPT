import torch
import torch.nn as nn

class Normalization(nn.Module):
    """
    Layer normalization module.

    Operates on the last dimension of the input tensor x, representing the embedding dimension.
    Scale and shift are trainable parameters.

    Args:
        emb_dim: Embedding dimension.

    Returns:
        torch.Tensor: Output tensor.
    """
    def __init__(self, emb_dim):
        super().__init__()

        self._eps = 1e-5  # Epsilon for numerical stability
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))


    def forward(self, x):
        """
        Forward pass for the layer normalization module.

        Computes the mean and variance of the input tensor x along the last dimension.
        Then normalizes the input tensor using the mean and variance.
        Finally, applies the scale and shift parameters to the normalized input tensor.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)  # GPT2's model uses unbiased=False
        norm_x = (x - mean) / torch.sqrt(variance + self._eps)
        return self.scale * norm_x + self.shift