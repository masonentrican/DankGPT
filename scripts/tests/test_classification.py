"""
Test classification fine tuning on spam dataset.
"""

import sys
import time
from matplotlib import pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from config.paths import DATA_DIR, MODELS_DIR, PROJECT_ROOT, SCRIPTS_DIR
from config.models import GPT2_SMALL
from config.training import QUICK
from llm import GPTModel, get_device, get_tokenizer
from llm.data.classification import ClassificationDataset
from llm.training.trainer import calc_classification_accuracy_loader, calc_classification_loss_loader, train_classification_model_simple
from llm.utils.classification import balance_two_class_dataset, train_val_test_split
from llm.utils.logging import get_logger, setup_logging
from llm.utils.weights import load_openai_weights_into_gpt

setup_logging()
logger = get_logger(__name__)

_data_file_path = DATA_DIR / "raw" / "sms_spam_collection.tsv" / "sms_spam_collection.tsv"
_train_cfg = QUICK.copy()

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
    # Use drop_rate=0.0 for classification (no dropout)
    model_cfg = GPT2_SMALL.copy()
    model_cfg["drop_rate"] = 0.0
    
    assert train_dataset.max_length < model_cfg['context_length'], (
        f"Train dataset max length {train_dataset.max_length} is greater than model context length {model_cfg['context_length']}. "
        "Reinitialize the dataset with a smaller max length."
    )

    # Prepare the model for training
    model = GPTModel(model_cfg)
    load_openai_weights_into_gpt(model, params)
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Create output head AFTER freezing (new parameters are trainable by default)
    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=model_cfg['emb_dim'], out_features=num_classes)
    
    # Create device
    device = get_device()
    logger.info(f"Device: {device}")

    # Load existing clasifier weights if they exist
    model_path = MODELS_DIR / "spam_classifier.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.info(f"Loaded existing classifier weights from {model_path}")
    else:
        logger.info(f"No existing classifier weights found at {model_path}")

    # Move to device
    model.to(device)
    
    # Unfreeze last transformer block and final LayerNorm
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
    torch.manual_seed(123)

    # optimizer will skip frozen params automatically
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    # Override basic training configuration for classification on small dataset
    _train_cfg["num_epochs"] = 5
    _train_cfg["eval_freq"] = 50
    _train_cfg["eval_iter"] = 5

    start_time = time.time()

    train_losses, val_losses, train_accuracies, val_accuracies, examples_seen = train_classification_model_simple(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        train_config=_train_cfg,
        tokenizer=tokenizer
    )

    end_time = time.time()
    logger.info(f"Training time: {(end_time - start_time) / 60} minutes")

    # Plot losses
    epochs_tensor = torch.linspace(0, _train_cfg["num_epochs"], len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss")

    # Plot accuracies
    epochs_tensor = torch.linspace(0, _train_cfg["num_epochs"], len(train_accuracies))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accuracies))
    plot_values(epochs_tensor, examples_seen_tensor, train_accuracies, val_accuracies, label="accuracy")

    # Compute full training validation test and accuracy
    train_accuracy = calc_classification_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_classification_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_classification_accuracy_loader(test_loader, model, device)
    
    logger.info(f"Train Accuracy: {train_accuracy*100:.2f}%")
    logger.info(f"Validation Accuracy: {val_accuracy*100:.2f}%")
    logger.info(f"Test Accuracy: {test_accuracy*100:.2f}%")

    # Classify Spam
    logger.info("------- Classify Spam -------")
    text = "Congratulations! You've won a $1000 Amazon gift card. Click the link to claim your prize."
    classified_label = classify_spam(text, model, tokenizer, device, max_length=model.context_length)
    logger.info(f"Text: {text}")
    logger.info(f"Classified Label: {classified_label}")

    text_2 = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
    classified_label2 = classify_spam(text_2, model, tokenizer, device, max_length=model.context_length)
    logger.info(f"Text: {text_2 }")
    logger.info(f"Classified Label: {classified_label2}")

    text_3 = "Congratulations! Now that you've got your promotion, can I get $5"
    classified_label3 = classify_spam(text_3, model, tokenizer, device, max_length=model.context_length)
    logger.info(f"Text: {text_3}")
    logger.info(f"Classified Label: {classified_label3}")

    text_4 = "Hey John, You're invited to a private crypto group led by industry leaders. Sign up here to get access to the group for trading calls."
    classified_label4 = classify_spam(text_4, model, tokenizer, device, max_length=model.context_length)
    logger.info(f"Text: {text_4}")
    logger.info(f"Classified Label: {classified_label4}")

    # Save Model
    model_path = MODELS_DIR / "spam_classifier.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")

    # End Test
    logger.info(50 * "=")
    logger.info("End Classification Test")
    logger.info(50 * "=" + "\n")

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation Loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples Seen")
    
    fig.tight_layout()

    # Ensure charts directory exists (relative to project root)
    charts_dir = PROJECT_ROOT / "charts"
    charts_dir.mkdir(exist_ok=True)

    plt.savefig(charts_dir / f"classification_{label}.png", dpi=150)
    logger.info(f"Saved plot to {charts_dir / f'classification_{label}.png'}")

def classify_spam(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.emb.pos_emb.weight.shape[0]

    # Truncate sequences if they are too long
    input_ids = input_ids[:min(max_length, supported_context_length)]
    assert max_length is not None, (
        "max_length must be specified. If you want to use the full model context, "
        "pass max_length=model.emb.pos_emb.weight.shape[0]."
    )
    assert max_length <= supported_context_length, (
        f"max_length ({max_length}) exceeds model's supported context length ({supported_context_length})."
    )
    
    # Store original length before padding
    original_length = len(input_ids)
    
    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        # Get logits for all tokens
        all_logits = model(input_tensor)  # [batch_size, seq_len, num_classes]
        # Use the last non-padding token (original_length - 1 because of 0-indexing)
        logits = all_logits[:, original_length - 1, :]  # Logits of the last actual token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"


if __name__ == "__main__":
    main()