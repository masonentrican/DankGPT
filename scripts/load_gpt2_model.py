"""
Download and load GPT-2 model from openai.
"""
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2("124M", "models/gpt2")

print("Settings:", settings)
print("Param dict keys:", params.keys())