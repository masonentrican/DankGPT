import urllib.request
from pathlib import Path

from llm.utils.logging import get_logger

logger = get_logger(__name__)

def download_text(url: str, dest_dir: Path, file_name: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / file_name

    logger.info(f"Downloading to {file_path}")
    urllib.request.urlretrieve(url, file_path)

    return file_path

