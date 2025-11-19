"""
Test script to demonstrate loading and using model configurations.
"""

from llm.config import load_config, list_available_configs

def main():
    print("Available configs:", list_available_configs())
    print()
    
    # Load the 124M config
    config = load_config("gpt2_small")
    
    print("GPT-2 124M Configuration:")
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

