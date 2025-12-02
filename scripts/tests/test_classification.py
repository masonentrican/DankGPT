"""
Test classification fine tuning on spam dataset.
"""

import pandas as pd
from config.paths import DATA_DIR
from llm.utils.classification import balance_two_class_dataset, train_val_test_split
from llm.utils.logging import get_logger, setup_logging



setup_logging()
logger = get_logger(__name__)

_data_file_path = DATA_DIR / "raw" / "sms_spam_collection.tsv" / "sms_spam_collection.tsv"

def main():
    dataFrame = pd.read_csv(_data_file_path, sep="\t", header=None, names=["Label", "Text"])

    logger.info(50 * "=" )
    logger.info("Begin Classification Test:")
    logger.info(50 * "=" + "\n")

    # Unbalanced DataFrame
    logger.info(f"Unbalanced DataFrame Loaded Successfully")
    logger.info("--------------------------------")
    logger.debug(f"DataFrame at path: {_data_file_path}")
    logger.debug(f"DataFrame Labels:\n\n{dataFrame['Label'].value_counts()}\n")

    # Balanced DataFrame
    balanced_df = balance_two_class_dataset(dataFrame)
    logger.info(f"Balanced DataFrame Created Successfully")
    logger.info("--------------------------------")
    logger.debug(f"DataFrame Labels: \n\n{balanced_df['Label'].value_counts()}\n")
    logger.debug(f"DataFrame Contents:\n\n{balanced_df}\n")
    
    # Create Dataset by splitting into train, validation, and test sets
    train_df, val_df, test_df = train_val_test_split(balanced_df)

    # Train DataFrame
    logger.info(f"Train DataFrame Created Successfully")
    logger.info("--------------------------------")
    logger.debug(f"Train DataFrame Labels: \n\n{train_df['Label'].value_counts()}\n")
    logger.debug(f"Train DataFrame Contents:\n\n{train_df}\n")
    
    # Validation DataFrame
    logger.info(f"Validation DataFrame Created Successfully")
    logger.info("--------------------------------")
    logger.debug(f"Validation DataFrame Labels: \n\n{val_df['Label'].value_counts()}\n")
    logger.debug(f"Validation DataFrame Contents:\n\n{val_df}\n")

    # Test DataFrame
    logger.info(f"Test DataFrame Created Successfully")
    logger.info("--------------------------------")
    logger.debug(f"Test DataFrame Labels: \n\n{test_df['Label'].value_counts()}\n")
    logger.debug(f"Test DataFrame Contents:\n\n{test_df}\n")

    # Check directory exist first, if not create it, then save dataframes
    (DATA_DIR / "processed" / "spam").mkdir(parents=True, exist_ok=True)
    train_df.to_csv((DATA_DIR / "processed" / "spam" / "train.csv"), index=False)
    val_df.to_csv((DATA_DIR / "processed" / "spam" / "val.csv"), index=False)
    test_df.to_csv((DATA_DIR / "processed" / "spam" / "test.csv"), index=False)

    logger.info(f"Saved DataFrames to: {DATA_DIR / 'processed' / 'spam'}\n")

    # End Test
    logger.info(50 * "=")
    logger.info("End Classification Test")
    logger.info(50 * "=" + "\n")
if __name__ == "__main__":
    main()