from pathlib import Path
from llm.data.importer import download_text
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
FILE_NAME = "the-verdict.txt"

file_path = download_text(URL, RAW_DIR, FILE_NAME)

print("Done! File downloaded:", file_path)
