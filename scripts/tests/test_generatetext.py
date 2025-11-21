"""
Test script for text generation using the GPT model.
"""

import torch
import tiktoken

from pathlib import Path
from torch.utils.data import DataLoader
from llm.data.dataset import GPTDataset
from llm.models.gptmodel import GPTModel

from llm.config import SMOOTHBRAIN

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text using greedy decoding (always picks the most likely token).
    """
    for _ in range(max_new_tokens):
        # Slice the context to fit within the model's context window
        # This keeps only the most recent 'context_size' tokens
        idx_cond = idx[:, -context_size:]
        
        # Forward pass: get logits for next token prediction
        # torch.no_grad() disables gradient computation for efficiency
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Extract logits for the last position (next token prediction)
        logits = logits[:, -1, :]
        
        # Convert logits to probabilities using softmax
        probas = torch.softmax(logits, dim=-1)
        
        # Greedy decoding: select the token with highest probability
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        # Append the new token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate the loss for a batch of input and target tokens using cross entropy loss.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_load(data_loader, model, device, num_batches=None):
    """
    Calculate the loss for a batch of input and target tokens using cross entropy loss.
    """
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader) # All batches if no number of batches is specified
    else:
        num_batches = min(num_batches, len(data_loader)) # match data loader length if num_batches is higher
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate the model on the train and validation sets.
    """
    model.eval() # Disable dropout and batch normalization
    with torch.no_grad():
        train_loss = calc_loss_load(train_loader, model, device, eval_iter) # Calculate loss on train set
        val_loss = calc_loss_load(val_loader, model, device, eval_iter) # Calculate loss on validation set
        model.train()
        return train_loss, val_loss

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Unsqueeze to add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # Squeeze to remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval() # Disable dropout and batch normalization
    context_size = model.emb.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def create_dataloader(
    text: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader over GPT-style sliding-window token sequences.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], [] # Track losses
    tokens_seen, global_step = 0, 0 # Track tokens seen and global step

    for epoch in range(num_epochs):
        model.train()
        for epoch in range(num_epochs):
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad() # Reset loss gradient from previous batch iteration
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward() # Backpropagate the loss
                optimizer.step() # UPDATE MODEL PARAMETERS
                tokens_seen += input_batch.numel()
                global_step += 1

                # Evaluate model on validation set
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Epoch {epoch+1}, Step {global_step}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

def main():
    """
    Main function to test the text generation functionality.
    """

    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    raw_file = raw_dir / "the-verdict.txt"
    text_data = raw_file.read_text(encoding="utf-8")

    cfg = SMOOTHBRAIN
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print(f"Total characters: {total_characters}")
    print(f"Total tokens: {total_tokens}")

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    num_epochs = 10
    start_context = "Every effort moves you"
    print(f"INPUT: {start_context}")

    train_loader = create_dataloader(train_data, batch_size=8, max_length=4, stride=4)
    val_loader = create_dataloader(val_data, batch_size=8, max_length=4, stride=4)

    train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq=5, eval_iter=5, start_context=start_context, tokenizer=tokenizer)

if __name__ == "__main__":
    main()