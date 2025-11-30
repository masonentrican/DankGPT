import sys
import torch
from config.models import GPT2_SMALL
from config.paths import DATA_DIR, MODELS_DIR, SCRIPTS_DIR
from llm import GPTModel, calc_loss_load, create_dataloader, get_device, get_tokenizer
from llm.utils.weights import load_openai_weights_into_gpt

# Add scripts directory to path to import gpt_download
sys.path.insert(0, str(SCRIPTS_DIR))
from gpt_download import download_and_load_gpt2

def main():
    """
    Calculate the loss of the GPT-2 small model. Using the verdict dataset.
    """

    # Set seed and initialize tokenizer
    torch.manual_seed(123)
    tokenizer = get_tokenizer()

    # Load GPT-2 model params from OpenAI
    model_path = MODELS_DIR / "gpt2"
    settings, params = download_and_load_gpt2("124M", model_path)

    # Initialize GPT model
    gpt = GPTModel(GPT2_SMALL)
    gpt.eval()
    device = get_device("auto")

    # Load OpenAI weights into GPT model
    load_openai_weights_into_gpt(gpt, params)
    gpt.to(device) # Move model to cuda if available

    # Load text data
    raw_file = DATA_DIR / "raw" / "the-verdict.txt"
    text_data = raw_file.read_text(encoding="utf-8")
    
    # Train/validation ratio
    train_ratio = 0.75
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)

    # Create dataloaders
    train_loader = create_dataloader(
        train_data,
        batch_size=2, 
        max_length=GPT2_SMALL["context_length"], 
        stride=GPT2_SMALL["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader(
        val_data,
        batch_size=2,
        max_length=GPT2_SMALL["context_length"],
        stride=GPT2_SMALL["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    torch.manual_seed(123) # Again because of shuffling in the dataloader

    train_loss = calc_loss_load(train_loader, gpt, device)
    val_loss = calc_loss_load(val_loader, gpt, device)

    print(f"Train loss: {train_loss:.4f}")
    print(f"Val loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()