import pandas as pd
from config.paths import DATA_DIR
from llm.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

_data_file_path = DATA_DIR / "raw" / "sms_spam_collection.tsv" / "sms_spam_collection.tsv"

def main():
    dataFrame = pd.read_csv(_data_file_path, sep="\t", header=None, names=["Label", "Text"])
    logger.info(f"DataFrame at path: {_data_file_path}")
    logger.info(f"DataFrame Labels: {dataFrame['Label'].value_counts()}")
    logger.info(f"DataFrame Contents: \n{dataFrame}")

if __name__ == "__main__":
    main()