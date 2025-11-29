import tiktoken
import torch
import sys
from pathlib import Path
from llm.generation.generator import generate_text
from llm.utils.tokenization import text_to_token_ids, token_ids_to_text
from llm.utils.weights import load_openai_weights_into_gpt
from llm.models.gptmodel import GPTModel
from config.models import GPT2_SMALL

# Add scripts directory to path to import gpt_download
scripts_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(scripts_dir))
from gpt_download import download_and_load_gpt2

# Init torch and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")

# Load GPT-2 model from OpenAI
project_root = Path(__file__).resolve().parents[2]
model_dir = project_root / "models"
model_path = model_dir / "gpt2"
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