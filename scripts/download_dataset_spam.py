from pathlib import Path
import urllib.request
import zipfile
import os
from config.paths import DATA_DIR
from llm.utils.logging import get_logger, setup_logging

"""
Download the SMS Spam Collection dataset from the UCI Machine Learning Repository.
"""

logger = get_logger(__name__)
setup_logging()

_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

_zip_name = "sms_spam_collection.zip"
_zip_path = DATA_DIR / "raw" / _zip_name

_extract_name = "sms_spam_collection.tsv"
_extract_path = DATA_DIR / "raw" / _extract_name

_data_file_path = Path(_extract_path) / _extract_name


def download_and_unzip(url, zip_path, extract_path, data_file_path, extract_name):
    if data_file_path.exists():
        logger.info(f"Skipping download - File already exists at: {data_file_path}")
        return
    
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
            logger.info(f"Downloaded: {zip_path}")
    
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
        logger.info(f"Extracted: {extract_path}")

    original_file_path = Path(extract_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    logger.info(f"Renamed: {original_file_path} to {data_file_path}")
    logger.info(f"Done! File downloaded: {data_file_path}")

if __name__ == "__main__":    
    download_and_unzip(_url, _zip_path, _extract_path, _data_file_path, _extract_name)
