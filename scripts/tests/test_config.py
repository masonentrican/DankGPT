"""
Test script to demonstrate loading and using model configurations.
"""

from config.models import GPT2_SMALL, GPT2_MEDIUM, GPT2_LARGE, GPT2_XLARGE
from llm.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

def main():
    logger.info("Available configs: GPT2_SMALL, GPT2_MEDIUM, GPT2_LARGE, GPT2_XLARGE")
    logger.info("")
    
    # Use the GPT-2 Small config
    config = GPT2_SMALL
    
    logger.info("GPT-2 Small (124M) Configuration:")
    logger.info("-" * 40)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    # Access specific values
    logger.info(f"Vocabulary size: {config['vocab_size']}")
    logger.info(f"Embedding dimension: {config['emb_dim']}")
    logger.info(f"Number of layers: {config['num_layers']}")
    logger.info(f"Number of heads: {config['num_heads']}")

if __name__ == "__main__":
    main()

