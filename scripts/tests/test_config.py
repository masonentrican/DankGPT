"""
Test script to demonstrate loading and using model configurations.
"""

from llm.config import GPT2_SMALL, GPT2_MEDIUM, GPT2_LARGE, GPT2_XLARGE

def main():
    print("Available configs: GPT2_SMALL, GPT2_MEDIUM, GPT2_LARGE, GPT2_XLARGE")
    print()
    
    # Use the GPT-2 Small config
    config = GPT2_SMALL
    
    print("GPT-2 Small (124M) Configuration:")
    print("-" * 40)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Access specific values
    print(f"Vocabulary size: {config['vocab_size']}")
    print(f"Embedding dimension: {config['emb_dim']}")
    print(f"Number of layers: {config['num_layers']}")
    print(f"Number of heads: {config['num_heads']}")

if __name__ == "__main__":
    main()

