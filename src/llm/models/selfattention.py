import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Simple single-head self-attention layer.

    Args:
        dim_in:  Input embedding dimension.
        dim_out: Output embedding dimension (projected Q/K/V size).
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:

        super().__init__()
        self.d_in = dim_in
        self.d_out = dim_out

        # Trainable weight matricies mapping input -> query / key / value
        self.weight_query = nn.Parameter(torch.rand(dim_in,dim_out))
        self.weight_key   = nn.Parameter(torch.rand(dim_in,dim_out))
        self.weight_value = nn.Parameter(torch.rand(dim_in,dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape:
            [T, D_in]      single sequence
         or [B, T, D_in]   batch of sequences

        Returns:
            context vectors with same leading dims as x:
            [T, D_out] or [B, T, D_out]
        """

        # Compute query, key and value vectors.
        keys = x @ self.weight_key
        queries = x @ self.weight_query
        values = x @ self.weight_value

        # Compute attention score similarly. Maxtrix math in
        # place of dot-product between query and keys
        attention_scores = queries @ keys.T

        # Compute attention weights through scaling by sqrt(d_k) which
		# is the last dim of the keys.
        scale = math.sqrt(keys.size(-1))
        attention_weights = torch.softmax(attention_scores / scale, dim=-1)

        # Compute resulting context vector for self-attention layer
        context_vector = attention_weights @ values

        return context_vector

