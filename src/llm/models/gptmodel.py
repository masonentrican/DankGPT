import math
import torch
import torch.nn as nn

from llm.models.embeddings import GPTEmbedding
from llm.models.normalization import Normalization
from llm.models.transformer import Transformer

class GPTModel(nn.Module):
    """
    GPT-2 style language model.

    The model consists of an embedding layer, dropout, a series of transformer blocks, and a final layer norm and output head.
    The embedding layer is used to embed the input tokens into a vector space.
    The transformer blocks are used to process the input tokens.
    The final layer norm and output head are used to produce the output logits.

    Args:
        cfg: Configuration dictionary containing model hyperparameters.
    """
    def __init__(self, cfg):
        """
        Initializes the GPT-2 style language model.
        
        Args:
            cfg: Configuration dictionary with keys:
                - vocab_size: Vocabulary size
                - context_length: Maximum context window size
                - emb_dim: Embedding dimension
                - num_layers: Number of transformer layers
                - num_heads: Number of attention heads
                - drop_rate: Dropout rate
                - qkv_bias: Whether to use bias in Q/K/V projections
        """   
        super().__init__()
        
        # Store config for later access
        self.cfg = cfg
        
        # Extract and store config values with defaults (exposed as properties)
        self.vocab_size = cfg.get("vocab_size", 50257)
        self.context_length = cfg.get("context_length", 1024)
        self.emb_dim = cfg.get("emb_dim", 768)
        self.num_layers = cfg.get("num_layers", 12)
        drop_rate = cfg.get("drop_rate", 0.1)

        # Embedding layer with optional dropout
        self.emb = GPTEmbedding(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            emb_dim=self.emb_dim,
            drop_rate=drop_rate
        )

        # Series of Transformer blocks driven by the configuration's number of layers
        self.trf_blocks = nn.Sequential(
            *[Transformer(cfg)
                for _ in range(self.num_layers)]
        )

        # Final layer normalization and output head
        self.final_norm = Normalization(self.emb_dim)
        self.out_head = nn.Linear(self.emb_dim, self.vocab_size, bias=False)

    def forward(self, in_idx, return_hidden_states: bool = False):
        """
        Forward pass.

        Computes the token embeddings, positional embeddings, and applies dropout.
        Then passes the output through the transformer blocks and final layer norm.
        Returns logits with the linear output layer.

        Args:
            in_idx: Input token indices [batch_size, seq_len].
            return_hidden_states: If True, returns hidden states before output head.
                                 If False, returns logits.

        Returns:
            torch.Tensor: 
                - If return_hidden_states=False: Logits [batch_size, seq_len, vocab_size]
                - If return_hidden_states=True: Hidden states [batch_size, seq_len, emb_dim]
        """
        x = self.emb(in_idx)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        
        if return_hidden_states:
            return x
        
        logits = self.out_head(x)
        return logits
    
    def get_hidden_states(self, in_idx):
        """
        Get hidden states from the model before the output head.
        
        Useful for fine-tuning tasks that need to add task-specific heads.
        
        Args:
            in_idx: Input token indices [batch_size, seq_len].
        
        Returns:
            torch.Tensor: Hidden states [batch_size, seq_len, emb_dim].
        """
        return self.forward(in_idx, return_hidden_states=True)