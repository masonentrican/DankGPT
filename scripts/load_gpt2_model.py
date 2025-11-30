"""
Download and load GPT-2 model from openai.
"""
from config.paths import MODELS_DIR
from gpt_download import download_and_load_gpt2

model_path = MODELS_DIR / "gpt2"

print(model_path)

settings, params = download_and_load_gpt2("1558M", model_path)

print("Settings:", settings)
print("Param dict keys:", params.keys())