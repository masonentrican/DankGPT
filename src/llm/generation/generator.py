"""
Text generation utilities for GPT models.
"""

import torch

from llm.utils.tokenization import text_to_token_ids, token_ids_to_text


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text using greedy decoding (always picks the most likely token).

    Args:
        model: GPT model instance.
        idx: Input token indices tensor with shape (batch_size, seq_len).
        max_new_tokens: Maximum number of new tokens to generate.
        context_size: Maximum context window size.

    Returns:
        torch.Tensor: Generated token indices with shape (batch_size, seq_len + max_new_tokens).
    """
    for _ in range(max_new_tokens):
        # Slice the context to fit within the model's context window
        # This keeps only the most recent 'context_size' tokens
        idx_cond = idx[:, -context_size:]
        
        # Forward pass: get logits for next token prediction
        # torch.no_grad() disables gradient computation for efficiency
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Extract logits for the last position (next token prediction)
        logits = logits[:, -1, :]
        
        # Convert logits to probabilities using softmax
        probas = torch.softmax(logits, dim=-1)
        
        # Greedy decoding: select the token with highest probability
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        # Append the new token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx


def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    Generate and print a text sample from the model.

    Args:
        model: GPT model instance.
        tokenizer: Tokenizer object.
        device: Device to run inference on.
        start_context: Starting text context for generation.
    """
    model.eval()  # Disable dropout and batch normalization
    context_size = model.emb.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

