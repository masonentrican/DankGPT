import math
import torch
import torch.nn as nn

from llm.utils.logging import get_logger

logger = get_logger(__name__)

class MultiHeadAttention(nn.Module):
    """
    Multi-head-attention.

    This module computes attention over a sequence using multiple heads.
    Each head has its own query, key, and value projections.
    The heads are combined and projected to the output dimension.

    Args:
        dim_in: Input embedding dimension.
        dim_out: Output embedding dimension (query/key/value projections).
        context_length: Maximum sequence length for which the causal mask is built.
        dropout: Dropout probability applied to attention weights.
        num_heads: Number of attention heads.
        qkv_bias: Whether to include bias terms in Q/K/V projections.
    """
    def __init__(self,
                dim_in: int,
                dim_out: int,
                context_length: int,
                dropout: float = 0,
                num_heads: int = 1,
                qkv_bias: bool = False):
        
        super().__init__()
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"

        self._dim_out = dim_out
        self._num_heads = num_heads
        self._head_dim = dim_out // num_heads  # Dimension per head (dim_out divided among heads)
        self.weight_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.weight_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.weight_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.out_projection = nn.Linear(dim_out, dim_out) # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
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
        Compute multi-head attention over the input sequence.

        Args:
            x: Input tensor of shape [batch_size, num_tokens, dim_in].

        Returns:
            Tensor of shape [batch_size, num_tokens, dim_out] representing
            the attention-weighted combination of value vectors.
        """
        b, num_tokens, dim_in = x.shape
        
        # Tensor shape: [(b)atch_size, num_(t)okens, (d)im_out]
        keys = self.weight_key(x)      
        queries = self.weight_query(x) 
        values = self.weight_value(x)   
        
        # Split dim_out into num_heads and head_dim by reshaping:
        # (b, num_tokens, dim_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self._num_heads, self._head_dim)
        values = values.view(b, num_tokens, self._num_heads, self._head_dim)
        queries = queries.view(b, num_tokens, self._num_heads, self._head_dim)

        # Transpose to move num_heads before num_tokens for batched attention computation:
        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores for each head:
        # (b, num_heads, num_tokens, head_dim) @ (b, num_heads, head_dim, num_tokens)
        # -> (b, num_heads, num_tokens, num_tokens)
        attention_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask so future positions are not attended to.
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        # Scale by sqrt(d_k) for stable gradients.
        scale = keys.shape[-1]**0.5
        attention_weights = torch.softmax(attention_scores / scale, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values:
        # (b, num_heads, num_tokens, num_tokens) @ (b, num_heads, num_tokens, head_dim)
        # -> (b, num_heads, num_tokens, head_dim)
        # Transpose to move num_heads dimension: -> (b, num_tokens, num_heads, head_dim)
        context_vector = (attention_weights @ values).transpose(1, 2)

        # Reshape to combine heads: (b, num_tokens, num_heads, head_dim) -> (b, num_tokens, dim_out)
        context_vector = context_vector.contiguous().view(b, num_tokens, self._dim_out)

        # Final projection to output dimension
        context_vector = self.out_projection(context_vector)

        return context_vector

class MultiHeadCausalAttention(nn.Module):
    """
    Multi-head causal self-attention.
    
    This module computes attention over a sequence where each position can
    only attend to current and previous positions (causal/auto-regressive).
    """
    def __init__(self, dim_in, dim_out, context_length: int, dropout: float = 0, num_heads: int = 1, qkv_bias: bool = False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                dim_in, dim_out, context_length, dropout, qkv_bias
                )
                for _ in range(num_heads)]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head causal attention over the input sequence.
        """
        return torch.cat([head(x) for head in self.heads], dim=-1)


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
        self._dim_out = dim_out
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
        logger.debug("-----------------------Causal Attention Test-----------------------")
        logger.debug(f"q:\n {self.weight_query.weight.T}")
        logger.debug(f"k:\n {self.weight_key.weight.T}")
        logger.debug(f"v:\n {self.weight_value.weight.T}")
        logger.debug(f"mask:\n {self.mask}")
        logger.debug(f"attention_weights:\n {attention_weights}")
        logger.debug(f"context_vector:\n {context_vector}")

        return context_vector

class SelfAttention(nn.Module):
    """
    Simple single-head self-attention layer.

    Args:
        dim_in:  Input embedding dimension.
        dim_out: Output embedding dimension (projected Q/K/V size).
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:

        super().__init__()
        self._d_in = dim_in
        self._d_out = dim_out

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
        logger.debug("-----------------------V1 TEST-----------------------")
        logger.debug(f"keys:\n {keys}")
        logger.debug(f"queries:\n {queries}")
        logger.debug(f"values:\n {values}")
        logger.debug(f"attention_scores:\n {attention_scores}")
        logger.debug(f"attention_weights:\n {attention_weights}")
        logger.debug(f"context_vector:\n {context_vector}")

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
        logger.debug("-----------------------V2 TEST-----------------------")
        logger.debug(f"keys:\n {keys}")
        logger.debug(f"queries:\n {queries}")
        logger.debug(f"values:\n {values}")
        logger.debug(f"attention_scores:\n {attention_scores}")
        logger.debug(f"attention_weights:\n {attention_weights}")
        logger.debug(f"context_vector:\n {context_vector}")

        return context_vector
