import torch
import torch.nn as nn

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
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        
        # Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[PlaceholderTransformerBlock(cfg) # TODO: Replace with actual transformer block
                for _ in range(cfg.n_layers)]
        )

        # Final layer norm and output head
        self.final_norm = nn.LayerNorm(cfg.emb_dim) #TODO: Replace with actual layer norm
        self.out.head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

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
        logits = self.out.head(x)
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

 

class PlaceholderLayerNorm(nn.Module):
    """
    Placeholder layer norm.

    This layer norm is a placeholder for the actual layer norm.
    It is used to test the training and inference pipeline.

    Args:
        cfg: Configuration dictionary.

    Returns:
        torch.Tensor: Output tensor.
    """
    def __init__(self, cfg): # TODO: Replace with actual layer norm
        super().__init__()

    def forward(self, x): # TODO: Replace with actual layer norm forward pass
        return x