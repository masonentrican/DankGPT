import torch

from config.models import SMOOTHBRAIN
from config.paths import DATA_DIR, MODELS_DIR
from llm.data.loader import create_dataloader
from llm.models.gptmodel import GPTModel
from llm.training import train_model_simple
from llm.utils import get_tokenizer
from llm.utils.plot import plot_losses

def main():
    """
    Main function to test the training and validation functionality.
    """
    
    # ============================================================================
    # Configuration
    # ============================================================================
    torch.manual_seed(123)
    cfg = SMOOTHBRAIN
    
    # Training parameters
    start_context = "Every effort moves you"
    train_ratio = 0.90
    num_epochs = 1
    batch_size = 8
    max_length = 4
    stride = 4

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_file.exists():
        # Load existing checkpoint
        print(f"Loading checkpoint from {model_file}")
        checkpoint = torch.load(model_file)
        model = GPTModel(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        # Initialize fresh model and optimizer
        print(f"No checkpoint found at {model_file}, starting fresh")
        model = GPTModel(cfg)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    # ============================================================================
    # Dataset statistics
    # ============================================================================
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print(f"Total characters: {total_characters}")
    print(f"Total tokens: {total_tokens}")

    # ============================================================================
    # Split data into training and validation sets
    # ============================================================================
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    # ============================================================================
    # Configure training parameters
    # ============================================================================
    # Number of complete passes through the training dataset
    print(f"INPUT: {start_context}")
    train_loader = create_dataloader(train_data, batch_size=batch_size, max_length=max_length, stride=stride)
    val_loader = create_dataloader(val_data, batch_size=batch_size, max_length=max_length, stride=stride)

    # ============================================================================
    # Execute training loop
    # ============================================================================
    # Train the model: forward pass, compute loss, backpropagate, update weights
    # eval_freq=5: run validation every 5 training iterations
    # eval_iter=5: perform 5 validation iterations when evaluating
    # Returns loss history and token count for monitoring training progress
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs, 
        eval_freq=5, eval_iter=5, start_context=start_context, tokenizer=tokenizer
    )

    # ============================================================================
    # Save the model
    # ============================================================================
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "model_and_optimizer.pth"
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        }, 
        model_path
    )
    print(f"Saved model and optimizer to {model_path}")

    # ============================================================================
    # Plot losses
    # ============================================================================    
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses("training_losses.png", epochs_tensor, tokens_seen, train_losses, val_losses)

if __name__ == "__main__":
    main()