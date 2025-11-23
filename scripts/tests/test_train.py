import torch
import tiktoken
from pathlib import Path

from llm.config import SMOOTHBRAIN
from llm.data.loader import create_dataloader
from llm.models.gptmodel import GPTModel
from llm.training import train_model_simple
from llm.utils.plot import plot_losses

def main():
    """
    Main function to test the training and validation functionality.
    """

    # ============================================================================
    # Load raw text data from file
    # ============================================================================
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    raw_file = raw_dir / "the-verdict.txt"
    text_data = raw_file.read_text(encoding="utf-8")

    # ============================================================================
    # Initialize model configuration and training components
    # ============================================================================
    # Load the model configuration (defines architecture: layers, heads, dimensions, etc.)
    cfg = SMOOTHBRAIN
    torch.manual_seed(123)

    # Initialize tokenizer to convert text to/from token IDs (using GPT-2 encoding)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the GPT model instance with the specified configuration
    model = GPTModel(cfg)
    model.to(device)

    # Initialize optimizer to update model weights during training
    # AdamW with learning rate 0.0004 and weight decay 0.1 for regularization
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
    # Reserve 90% of data for training, 10% for validation
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    
    # Split text
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    # ============================================================================
    # Configure training parameters
    # ============================================================================
    # Number of complete passes through the training dataset
    num_epochs = 10
    
    start_context = "Every effort moves you"
    print(f"INPUT: {start_context}")

    # ============================================================================
    # Create data loaders for batching
    # ============================================================================
    # Create data loaders that split text into fixed-length sequences
    # batch_size=8: process 8 sequences simultaneously
    # max_length=4: each sequence is 4 tokens long
    # stride=4: non-overlapping sequences (move 4 tokens forward each time)
    train_loader = create_dataloader(train_data, batch_size=8, max_length=4, stride=4)
    val_loader = create_dataloader(val_data, batch_size=8, max_length=4, stride=4)

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
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "model_and_optimizer.pth"
    
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