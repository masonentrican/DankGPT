"""
Text generation utilities for GPT models.
"""

import torch

from llm.utils.tokenization import text_to_token_ids, token_ids_to_text


def generate_text(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Generate text using sampling (greedy decoding or top-k sampling with temperature scaling).

    Args:
        model: GPT model instance.
        idx: Input token indices tensor with shape (batch_size, seq_len).
        max_new_tokens: Maximum number of new tokens to generate.
        context_size: Maximum context window size.
        temperature: Temperature for temperature scaling.
        top_k: Top-k sampling parameter.
        eos_id: End-of-sequence token ID.

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

        # Apply top-k sampling (filter out low probability tokens)
        if top_k is not None:
            top_k_logits, _ = torch.topk(logits, top_k)
            min_val = top_k_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            # Numerical stability tip to get equivalent results on mps devices
            logits = logits - logits.max(dim=-1, keepdim=True).values
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Stop generating early if end-of-sequence token is encountered and eos_id is specified
        if idx_next == eos_id:
            break

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)
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
        token_ids = generate_text(
            model,
            encoded,
            max_new_tokens=15,
            context_size=context_size,
            top_k=25,
            temperature=1.4
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

