"""
Training and evaluation utilities for GPT models.
"""

import torch

from llm.generation import generate_and_print_sample


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate the loss for a batch of input and target tokens using cross entropy loss.

    Args:
        input_batch: Input token indices tensor.
        target_batch: Target token indices tensor.
        model: GPT model instance.
        device: Device to run computation on.

    Returns:
        torch.Tensor: Loss value.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_load(data_loader, model, device, num_batches=None):
    """
    Calculate the average loss over multiple batches from a DataLoader.

    Args:
        data_loader: DataLoader providing input and target batches.
        model: GPT model instance.
        device: Device to run computation on.
        num_batches: Number of batches to evaluate. If None, evaluates all batches.

    Returns:
        float: Average loss value.
    """
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)  # All batches if no number of batches is specified
    else:
        num_batches = min(num_batches, len(data_loader))  # match data loader length if num_batches is higher
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

    Args:
        model: GPT model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to run computation on.
        eval_iter: Number of batches to evaluate on.

    Returns:
        tuple: (train_loss, val_loss)
    """
    model.eval()  # Disable dropout and batch normalization
    with torch.no_grad():
        train_loss = calc_loss_load(train_loader, model, device, eval_iter)  # Calculate loss on train set
        val_loss = calc_loss_load(val_loader, model, device, eval_iter)  # Calculate loss on validation set
    model.train()
    return train_loss, val_loss


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """
    Simple training loop for GPT model.

    Args:
        model: GPT model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer instance.
        device: Device to run training on.
        num_epochs: Number of training epochs.
        eval_freq: Frequency of evaluation (every N steps).
        eval_iter: Number of batches to evaluate on.
        start_context: Starting context for text generation samples.
        tokenizer: Tokenizer object.

    Returns:
        tuple: (train_losses, val_losses, track_tokens_seen)
    """
    train_losses, val_losses, track_tokens_seen = [], [], []  # Track losses
    tokens_seen, global_step = 0, 0  # Track tokens seen and global step

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradient from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # UPDATE MODEL PARAMETERS
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

