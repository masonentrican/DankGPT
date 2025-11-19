"""
Test script for text generation using the GPT model.

This script tests the basic text generation functionality by:
1. Loading a GPT model configuration
2. Encoding an initial context string
3. Generating new tokens using greedy decoding
4. Decoding and displaying the generated text
"""

import torch
import tiktoken
from llm.config import load_config
from llm.models.gptmodel import GPTModel


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


# ============================================================================
# Test Configuration and Setup
# ============================================================================

# Load model configuration
cfg = load_config("gpt2_small")

# Initialize tokenizer (GPT-2 BPE encoding)
tokenizer = tiktoken.get_encoding("gpt2")

# Initialize the GPT model
model = GPTModel(cfg)

# Set model to evaluation mode (disables dropout, batch norm updates, etc.)
model.eval()

# ============================================================================
# Text Generation Test
# ============================================================================

# Define the starting context/prompt
start_context = "Hello, I am"
print(f"INPUT: {start_context}")

# Encode the text into token IDs
encoded = tokenizer.encode(start_context)

# Convert to tensor and add batch dimension: (seq_len,) -> (1, seq_len)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

# Generate new tokens
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=9,
    context_size=cfg["context_length"]
)

# Decode the generated tokens back to text
decoded_text = tokenizer.decode(out[0].tolist())
print(f"DANK GPT SAYS: {decoded_text}")
