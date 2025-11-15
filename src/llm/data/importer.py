import urllib.request
from pathlib import Path

def download_text(url: str, dest_dir: Path, file_name: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / file_name

    print(f"Downloading to {file_path}")
    urllib.request.urlretrieve(url, file_path)

    return file_path

