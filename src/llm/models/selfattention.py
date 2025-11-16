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

        # Trainable weight matrices mapping input -> query / key / value
        self.weight_query = nn.Parameter(torch.rand(dim_in,dim_out))
        self.weight_key   = nn.Parameter(torch.rand(dim_in,dim_out))
        self.weight_value = nn.Parameter(torch.rand(dim_in,dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute self-attention for a single sequence.

        Note:
            This implementation assumes a 2D input [T, D_in]. For batched
            inputs, use a batched variant that handles [B, T, D_in].

        Args:
            x: Input tensor of shape [T, D_in].

        Returns:
            Context vectors of shape [T, D_out].
        """

        # Compute query, key and value vectors.
        keys = x @ self.weight_key
        queries = x @ self.weight_query
        values = x @ self.weight_value

        # Compute attention scores via matrix multiplication
        # (equivalent to pairwise dot-products between queries and keys).
        attention_scores = queries @ keys.T

        # Scale by sqrt(d_k) (last dimension of keys) for stable gradients.
        scale = math.sqrt(keys.size(-1))
        attention_weights = torch.softmax(attention_scores / scale, dim=-1)

        # Attention-weighted sum of values yields contextualized representations.
        context_vector = attention_weights @ values

        # Debug prints
        print("-----------------------V1 TEST-----------------------")
        print("keys: ", keys)
        print("queries: ", queries)
        print("values: ", values)
        print("attention_scores: ", attention_scores)
        print("attention_weights: ", attention_weights)
        print("context_vector: ", context_vector)



        return context_vector


class SelfAttentionV2(nn.Module):
    """
    Enhanced to use PyTorch nn.Linear layers for more efficient matrix
    multiplication (when bias units are disabled). Uses an optimized
    weight initialization scheme for more stable and effective training.

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
        """
        Compute self-attention for a single sequence [T, D_in].

        Note:
            This implementation assumes a 2D input [T, D_in]. For batched
            inputs, use a batched variant that handles [B, T, D_in].
        """
        # Compute query / key / value projections and attention scores
        keys = self.weight_key(x)
        queries = self.weight_query(x)
        values = self.weight_value(x)
        attention_scores = queries @ keys.T

        # Scale by sqrt(d_k) (last dimension of keys) for stable gradients.
        scale = math.sqrt(keys.size(-1))
        attention_weights = torch.softmax(attention_scores / scale, dim=-1)

        context_vector = attention_weights @ values

        # Debug prints
        print("-----------------------V2 TEST-----------------------")
        print("keys: ", keys)
        print("queries: ", queries)
        print("values: ", values)
        print("attention_scores: ", attention_scores)
        print("attention_weights: ", attention_weights)
        print("context_vector: ", context_vector)

        return context_vector
