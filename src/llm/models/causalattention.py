import torch
import torch.nn as nn
import math

class CausalAttention(nn.Module):
    """
    Single-head causal self-attention.
    
    This module computes attention over a sequence where each position can
    only attend to current and previous positions (causal/auto-regressive).
    
    Args:
        dim_in: Input embedding dimension.
        dim_out: Output embedding dimension (query/key/value projections).
        context_length: Maximum sequence length for which the causal mask is built.
        dropout: Dropout probability applied to attention weights.
        qkv_bias: Whether to include bias terms in Q/K/V projections (unused here).
    """
    def __init__(self,
                dim_in,
                dim_out,
                context_length: int,
                dropout: float = 0,
                qkv_bias: bool = False):

        super().__init__()
        self.dim_out = dim_out
        self.weight_query = nn.Linear(dim_in, dim_out)
        self.weight_key = nn.Linear(dim_in, dim_out)
        self.weight_value = nn.Linear(dim_in, dim_out)
        self.dropout = nn.Dropout(dropout)
        # Upper-triangular (excluding diagonal) mask enforces causality:
        # positions can attend only to themselves and earlier tokens.
        self.register_buffer(
            'mask', 
            torch.triu(
                torch.ones(
                    context_length,
                    context_length
                ),
                diagonal=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute causal attention over the input sequence.

        Args:
            x: Input tensor of shape [batch_size, num_tokens, dim_in].

        Returns:
            Tensor of shape [batch_size, num_tokens, dim_out] representing
            the attention-weighted combination of value vectors.
        """
        b, num_tokens, dim_in = x.shape

        # Linear projections for queries, keys, and values: [B, T, D_out]
        keys = self.weight_key(x)
        queries = self.weight_query(x)
        values = self.weight_value(x)
        
        # Raw attention scores: [B, T, T]
        # (Queries attend over all keys within the same sequence.)
        attention_scores = queries @ keys.transpose(1, 2)
        # Apply causal mask so future positions are not attended to.
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
            )
        
        # Scale by sqrt(d_k) for stable gradients (as in "Attention is All You Need").
        scale = math.sqrt(keys.size(-1))
        attention_weights = torch.softmax(attention_scores / scale, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum of values gives the contextualized representations.
        context_vector = attention_weights @ values

        # Debug prints
        print("-----------------------Causal Attention Test-----------------------")
        print("q: ", self.weight_query.weight.T)
        print("k: ", self.weight_key.weight.T)
        print("v: ", self.weight_value.weight.T)
        print("mask: ", self.mask)
        print("attention_weights: ", attention_weights)
        print("context_vector: ", context_vector)

        return context_vector