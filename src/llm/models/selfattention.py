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

        torch.Tensor

        print("q: ", self.weight_query)
        print("k: ", self.weight_key)
        print("v: ", self.weight_value)



        return context_vector


class SelfAttentionV2(nn.Module):
    """
    Enhanced to use pytorch nn.Linear layers to perform more efficient
    matrix multiplcation whe bias units are disabled. Has optimized
    weight init scheme resulting in more stable and efftive training

    Args:
        dim_in:  Input embedding dimension.
        dim_out: Output embedding dimension (projected Q/K/V size).
    """

    def __init__(self, dim_in, dim_out, qkv_bias=False) -> None:
        super().__init__()
        self.weight_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.weight_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.weight_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute query / key / value vectors and attention score
        keys = self.weight_key(x)
        queries = self.weight_query(x)
        values = self.weight_value(x)
        attention_scores = queries @ keys.T

        # Compute attention weights through scaling by sqrt(d_k) which
        # is the last dim of the keys.
        scale = math.sqrt(keys.size(-1))
        attention_weights = torch.softmax(attention_scores / scale, dim=-1)

        context_vector = attention_weights @ values

        print("q: ", self.weight_query.weight.T)
        print("k: ", self.weight_key.weight.T)
        print("v: ", self.weight_value.weight.T)

        return context_vector
