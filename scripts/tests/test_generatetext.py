"""
Test script for text generation using the GPT model.
"""

import torch
import tiktoken
from llm.config import SMOOTHBRAIN
from llm.models.gptmodel import GPTModel

def main():
    """
    Main function to test the text generation functionality.
    """
    cfg = SMOOTHBRAIN
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    model = GPTModel(cfg)
    model.eval()

    # Define the starting context/prompt
    start_context = "Every effort moves you"
    print(f"INPUT: {start_context}")

    # Generate new tokens
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=cfg["context_length"]
    )

    print(f"DANK GPT SAYS: {token_ids_to_text(token_ids, tokenizer)}")

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text using greedy decoding (always picks the most likely token).
    
    Args:
        model: The GPT model to use for generation
        idx: Input token indices tensor of shape (batch_size, seq_len)
        max_new_tokens: Maximum number of new tokens to generate
        context_size: Maximum context window size to use for each forward pass
        
    Returns:
        Tensor of shape (batch_size, original_seq_len + max_new_tokens) containing
        the original input tokens plus the newly generated tokens
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

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Unsqueeze to add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # Squeeze to remove batch dimension
    return tokenizer.decode(flat.tolist())

if __name__ == "__main__":
    main()