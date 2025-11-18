import torch
import torch.nn as nn

from llm.config import load_config

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

        # Embedding layers
        self.tok_emb = nn.Embedding(cfg.get("vocab_size"), cfg.get("emb_dim"))
        self.pos_emb = nn.Embedding(cfg.get("context_length"), cfg.get("emb_dim"))
        self.drop_emb = nn.Dropout(cfg.get("drop_rate"))
        
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
        Forward pass for the placeholder GPT model.

        Describes the data flow through the model. Computes the token embeddings, positional
        embeddings, and applies dropout. Then passes the output through the transformer blocks
        and final layer norm. Finally, produces logits with the linear output layer.

        Args:
            in_idx: Input token indices.

        Returns:
            torch.Tensor: Output logits.
        """
        batch_size, seq_len = in_idx.shape
        
        # Embed tokens and positions
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # Combine embeddings and apply dropout
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        # Output head
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

 

class LayerNormalization(nn.Module):
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

        self.eps = 1e-5 # Epsilon for numerical stability
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
        variance = x.var(dim=-1, keepdim=True, unbiased=False) # GPT2's model uses unbiased=False
        norm_x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * norm_x + self.shift