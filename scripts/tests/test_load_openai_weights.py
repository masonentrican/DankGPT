import torch
import sys
from pathlib import Path
from llm.utils import get_device, get_tokenizer
from llm.generation.generator import generate_text
from llm.utils.tokenization import text_to_token_ids, token_ids_to_text
from llm.utils.weights import load_openai_weights_into_gpt
from llm.models.gptmodel import GPTModel
from config.models import GPT2_SMALL
from config.paths import PROJECT_ROOT, MODELS_DIR, SCRIPTS_DIR

# Add scripts directory to path to import gpt_download
sys.path.insert(0, str(SCRIPTS_DIR))
from gpt_download import download_and_load_gpt2

# Init torch and tokenizer
device = get_device("auto")
torch.manual_seed(123)
tokenizer = get_tokenizer()

# Load GPT-2 model from OpenAI
model_path = MODELS_DIR / "gpt2"
settings, params = download_and_load_gpt2("124M", model_path)

# Init GPT model
gpt = GPTModel(GPT2_SMALL)
gpt.eval()
load_openai_weights_into_gpt(gpt, params)
gpt.to(device) # Move model to cuda if available

# Generate text
token_ids = generate_text(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=GPT2_SMALL["context_length"],
    top_k=50,
    temperature=1.5
)

print("-----------------------Load OpenAI Weights Test-----------------------")
print("Generated text:\n", token_ids_to_text(token_ids, tokenizer))