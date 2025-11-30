from config.paths import DATA_DIR
from llm import download_text

RAW_DIR = DATA_DIR / "raw"
URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
FILE_NAME = "the-verdict.txt"

file_path = download_text(URL, RAW_DIR, FILE_NAME)

print("Done! File downloaded:", file_path)
