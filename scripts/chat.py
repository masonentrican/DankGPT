import argparse
import sys

import torch
from llm import GPTModel, generate_text, get_device, get_tokenizer
from llm.utils.tokenization import text_to_token_ids, token_ids_to_text
from llm.utils.weights import load_openai_weights_into_gpt
from llm.utils.logging import get_logger, setup_logging
from config.models import GPT2_XLARGE
from config.paths import MODELS_DIR, SCRIPTS_DIR

# Add scripts directory to path to import gpt_download
sys.path.insert(0, str(SCRIPTS_DIR))
from gpt_download import download_and_load_gpt2

logger = get_logger(__name__)

def main(gpt_config, input_prompt, model_size, device):

    settings, params = download_and_load_gpt2(model_size=model_size, models_dir=MODELS_DIR / "gpt2")

    gpt = GPTModel(gpt_config)
    load_openai_weights_into_gpt(gpt, params)
    gpt.to(device)
    gpt.eval()

    tokenizer = get_tokenizer()
    torch.manual_seed(123)

    token_ids = generate_text(
        model=gpt,
        idx=text_to_token_ids(input_prompt, tokenizer).to(device),
        max_new_tokens=250,
        context_size=gpt_config["context_length"],
        top_k=50,
        temperature=1.0
    )

    logger.info(f"Output text:\n{token_ids_to_text(token_ids, tokenizer)}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate text with a pretrained GPT-2 model.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for running inference, e.g., cpu, cuda, mps, or auto. Defaults to auto."
    )

    args = parser.parse_args()

    torch.manual_seed(123)

    setup_logging()
    
    INPUT_PROMPT = input("Enter your prompt: ")
    DEVICE = get_device(args.device)

    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Device: {DEVICE}")
    logger.info("")

    main(GPT2_XLARGE, INPUT_PROMPT, "1558M", DEVICE)