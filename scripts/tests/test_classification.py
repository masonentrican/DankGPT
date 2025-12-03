"""
Test classification fine tuning on spam dataset.
"""

import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from config.paths import DATA_DIR, MODELS_DIR, SCRIPTS_DIR
from config.models import GPT2_SMALL
from llm import GPTModel, generate_text, get_device, get_tokenizer
from llm.data.classification import ClassificationDataset
from llm.training.trainer import calc_classification_accuracy_loader
from llm.utils.classification import balance_two_class_dataset, train_val_test_split
from llm.utils.logging import get_logger, setup_logging
from llm.utils.tokenization import text_to_token_ids, token_ids_to_text
from llm.utils.weights import load_openai_weights_into_gpt

setup_logging()
logger = get_logger(__name__)

_data_file_path = DATA_DIR / "raw" / "sms_spam_collection.tsv" / "sms_spam_collection.tsv"

def main():
    dataFrame = pd.read_csv(_data_file_path, sep="\t", header=None, names=["Label", "Text"])

    logger.info(50 * "=" )
    logger.info("Begin Classification Test:")
    logger.info(50 * "=" + "\n")

    # Unbalanced DataFrame
    logger.info("------- Unbalanced DataFrame -------")
    logger.info(f"Unbalanced DataFrame Loaded Successfully")
    logger.debug(f"DataFrame at path: {_data_file_path}")
    logger.debug(f"DataFrame Labels:\n\n{dataFrame['Label'].value_counts()}\n")

    # Balanced DataFrame
    balanced_df = balance_two_class_dataset(dataFrame)
    logger.info("------- Balanced DataFrame -------")
    logger.info(f"Balanced DataFrame Created Successfully")
    logger.debug(f"DataFrame Labels: \n\n{balanced_df['Label'].value_counts()}\n")
    logger.debug(f"DataFrame Contents:\n\n{balanced_df}\n")

    # Map labels to integers for training. This is hardcoded to this specific dataset.
    # TODO: Fit this into a general function or better yet baked into balance_two_class_dataset()?
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    
    # Create Dataset by splitting into train, validation, and test sets
    train_df, val_df, test_df = train_val_test_split(balanced_df)

    # Train DataFrame
    logger.info("------- Train DataFrame -------")
    logger.info(f"Train DataFrame Created Successfully")
    logger.debug(f"Train DataFrame Labels: \n\n{train_df['Label'].value_counts()}\n")
    logger.debug(f"Train DataFrame Contents:\n\n{train_df}\n")
    
    # Validation DataFrame
    logger.info("------- Validation DataFrame -------")
    logger.info(f"Validation DataFrame Created Successfully")
    logger.debug(f"Validation DataFrame Labels: \n\n{val_df['Label'].value_counts()}\n")
    logger.debug(f"Validation DataFrame Contents:\n\n{val_df}\n")

    # Test DataFrame
    logger.info("------- Test DataFrame -------")
    logger.info(f"Test DataFrame Created Successfully")
    logger.debug(f"Test DataFrame Labels: \n\n{test_df['Label'].value_counts()}\n")
    logger.debug(f"Test DataFrame Contents:\n\n{test_df}\n")

    # Check directory exist first, if not create it, then save dataframes
    (DATA_DIR / "processed" / "spam").mkdir(parents=True, exist_ok=True)
    train_df.to_csv((DATA_DIR / "processed" / "spam" / "train.csv"), index=False)
    val_df.to_csv((DATA_DIR / "processed" / "spam" / "val.csv"), index=False)
    test_df.to_csv((DATA_DIR / "processed" / "spam" / "test.csv"), index=False)

    logger.info(f"Saved DataFrames to: {DATA_DIR / 'processed' / 'spam'}")

    # Create Datasets
    logger.info("------- Create Datasets -------")
    tokenizer = get_tokenizer()
    train_dataset = ClassificationDataset(
        csv_file=DATA_DIR / "processed" / "spam" / "train.csv",
        tokenizer=tokenizer,
    )

    val_dataset = ClassificationDataset(
        csv_file=DATA_DIR / "processed" / "spam" / "val.csv",
        tokenizer=tokenizer,
        max_length=train_dataset.max_length
    )

    test_dataset = ClassificationDataset(
        csv_file=DATA_DIR / "processed" / "spam" / "test.csv",
        tokenizer=tokenizer,
        max_length=train_dataset.max_length
    )

    logger.info(f"Train Dataset Valid: {train_dataset is not None}")
    logger.info(f"Validation Dataset Valid: {val_dataset is not None}")
    logger.info(f"Test Dataset Valid: {test_dataset is not None}")

    logger.info("------- Create DataLoaders -------")
    # Create DataLoaders
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )

    logger.info(f"Train Loader Valid: {train_loader is not None}")
    logger.info(f"Validation Loader Valid: {val_loader is not None}")
    logger.info(f"Test Loader Valid: {test_loader is not None}")
    
    for input_batch, target_batch in train_loader:
        pass
    logger.info(f"Input Batch Dimensions: {input_batch.shape}")
    logger.info(f"Target Batch Dimensions: {target_batch.shape}")

    logger.info(f"{len(train_loader)} train batches")
    logger.info(f"{len(val_loader)} validation batches")
    logger.info(f"{len(test_loader)} test batches")

    # Add scripts directory to path to import gpt_download
    logger.info("------- Retrieve GPT-2 Model -------")
    sys.path.insert(0, str(SCRIPTS_DIR))
    from gpt_download import download_and_load_gpt2
    model_path = MODELS_DIR / "gpt2"
    settings, params = download_and_load_gpt2("124M", model_path)

    # Train Model
    logger.info("------- Train Model -------")
    assert train_dataset.max_length < GPT2_SMALL['context_length'], (
        f"Train dataset max length {train_dataset.max_length} is greater than GPT2_SMALL context length {GPT2_SMALL['context_length']}. "
        "Reinitialize the dataset with a smaller max length."
    )

    # Prepare the model for training
    model = GPTModel(GPT2_SMALL)
    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=GPT2_SMALL['emb_dim'], out_features=num_classes)

    # Freeze training of the whole model except the last transformer block and final LayerNorm module
    for param in model.parameters():
        param.requires_grad = False
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    # Test the model
    logger.info("------- Test Model -------")

    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)

    logger.info(f"Inputs: {inputs}")
    logger.info(f"Inputs dimensions: {inputs.shape}")

    device = get_device()

    logger.info(f"Device: {device}")
    model.to(device)
    torch.manual_seed(123)

    train_accuracy = calc_classification_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_classification_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_classification_accuracy_loader(test_loader, model, device, num_batches=10)

    logger.info(f"Train Accuracy: {train_accuracy*100:.2f}%")
    logger.info(f"Validation Accuracy: {val_accuracy*100:.2f}%")
    logger.info(f"Test Accuracy: {test_accuracy*100:.2f}%")



    # End Test
    logger.info(50 * "=")
    logger.info("End Classification Test")
    logger.info(50 * "=" + "\n")
    
if __name__ == "__main__":
    main()