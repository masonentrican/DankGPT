"""
Download and load GPT-2 model from openai.
"""
from pathlib import Path
from gpt_download import download_and_load_gpt2

project_root = Path(__file__).resolve().parents[1]
model_dir = project_root / "models"
model_path = model_dir / "gpt2"

print(model_path)

settings, params = download_and_load_gpt2("124M", model_path)

print("Settings:", settings)
print("Param dict keys:", params.keys())