import torch

from config.models import SMOOTHBRAIN
from config.paths import DATA_DIR, MODELS_DIR
from config.training import QUICK
from llm import (
    GPTModel,
    create_dataloader,
    get_device,
    get_tokenizer,
    load_checkpoint,
    save_checkpoint,
    train_model_simple,
)
from llm.utils.plot import plot_losses
from llm.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

def main():
    """
    Main function to test the training and validation functionality.
    """
    
    # ============================================================================
    # Configuration
    # ============================================================================
    torch.manual_seed(123)
    model_cfg = SMOOTHBRAIN
    train_cfg = QUICK.copy()  # Use training config, can be modified if needed

    # ============================================================================
    # IO Bootstrap
    # ============================================================================
    raw_dir = DATA_DIR / "raw"
    model_dir = MODELS_DIR
    
    # Training data
    raw_file = raw_dir / "the-verdict.txt"
    text_data = raw_file.read_text(encoding="utf-8")
    
    # Model and optimizer checkpoint
    model_file = model_dir / "model_and_optimizer.pth"

    # ============================================================================
    # Start from previous model checkpoint if it exists, otherwise start fresh
    # ============================================================================

    tokenizer = get_tokenizer()
    device = get_device("auto")

    # Initialize model and optimizer
    model = GPTModel(model_cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )

    if model_file.exists():
        # Load existing checkpoint
        try:
            metadata = load_checkpoint(model_file, model, optimizer, device)
            logger.info(f"Resumed from checkpoint. Epoch: {metadata.get('epoch', 'N/A')}, Loss: {metadata.get('loss', 'N/A')}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}. Starting fresh.")
    else:
        logger.info(f"No checkpoint found at {model_file}, starting fresh")

    # ============================================================================
    # Dataset statistics
    # ============================================================================
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    logger.info(f"Total characters: {total_characters}")
    logger.info(f"Total tokens: {total_tokens}")

    # ============================================================================
    # Split data into training and validation sets
    # ============================================================================
    train_ratio = train_cfg["train_ratio"]
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    # ============================================================================
    # Create data loaders
    # ============================================================================
    logger.info(f"INPUT: {train_cfg['start_context']}")
    train_loader = create_dataloader(
        train_data,
        batch_size=train_cfg["batch_size"],
        max_length=train_cfg["max_length"],
        stride=train_cfg["stride"],
        shuffle=train_cfg.get("shuffle", True),
        drop_last=train_cfg.get("drop_last", True),
        num_workers=train_cfg.get("num_workers", 0),
    )
    val_loader = create_dataloader(
        val_data,
        batch_size=train_cfg["batch_size"],
        max_length=train_cfg["max_length"],
        stride=train_cfg["stride"],
        shuffle=False,  # Don't shuffle validation
        drop_last=False,  # Don't drop last batch in validation
        num_workers=train_cfg.get("num_workers", 0),
    )

    # ============================================================================
    # Execute training loop
    # ============================================================================
    # Train the model: forward pass, compute loss, backpropagate, update weights
    # Returns loss history and token count for monitoring training progress
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, train_cfg, tokenizer
    )

    # ============================================================================
    # Save the model
    # ============================================================================
    model_path = model_dir / "model_and_optimizer.pth"
    
    # Get final loss for saving
    final_val_loss = val_losses[-1] if val_losses else None
    
    save_checkpoint(
        model=model,
        filepath=model_path,
        optimizer=optimizer,
        epoch=train_cfg["num_epochs"],
        loss=final_val_loss,
    )

    # ============================================================================
    # Plot losses
    # ============================================================================    
    epochs_tensor = torch.linspace(0, train_cfg["num_epochs"], len(train_losses))
    plot_losses("training_losses.png", epochs_tensor, tokens_seen, train_losses, val_losses)

if __name__ == "__main__":
    main()