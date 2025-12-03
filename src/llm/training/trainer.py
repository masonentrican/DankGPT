"""
Training and evaluation utilities for GPT models.
"""

import torch
from typing import TYPE_CHECKING

from llm.generation import generate_and_print_sample
from llm.utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from config.training import TrainingConfig


def calc_language_loss_batch(input_batch, target_batch, model, device):
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
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_language_loss_loader(data_loader, model, device, num_batches=None):
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
            loss = calc_language_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def calc_classification_accuracy_loader(data_loader, model, device, num_batches=None, pad_token_id=50256):
    """
    Calculate the accuracy over multiple batches from a classification DataLoader.

    Args:
        data_loader: DataLoader providing input and target batches.
        model: GPT model instance.
        device: Device to run computation on.
        num_batches: Number of batches to evaluate. If None, evaluates all batches.
        pad_token_id: Token ID used for padding.

    Returns:
        float: Accuracy value.
    """
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                all_logits = model(input_batch)  # [batch_size, seq_len, num_classes]
                # Find last non-padding token for each sequence in the batch
                mask = (input_batch != pad_token_id)
                # Last non-padding index = number of non-padding tokens - 1
                last_non_pad_indices = (mask.sum(dim=1) - 1).clamp(min=0)
                # Extract logits for the last non-padding token of each sequence
                batch_indices = torch.arange(input_batch.size(0), device=device)
                logits = all_logits[batch_indices, last_non_pad_indices, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

def calc_classification_loss_batch(input_batch, target_batch, model, device, pad_token_id=50256):
    """
    Calculate the loss for a batch of input and target tokens using cross entropy loss.

    Args:
        input_batch: Input token indices tensor.
        target_batch: Target token indices tensor.
        model: GPT model instance.
        device: Device to run computation on.
        pad_token_id: Token ID used for padding.

    Returns:
        torch.Tensor: Loss value.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    all_logits = model(input_batch)  # [batch_size, seq_len, num_classes]
    # Find last non-padding token for each sequence in the batch
    mask = (input_batch != pad_token_id)
    # Last non-padding index = number of non-padding tokens - 1
    last_non_pad_indices = (mask.sum(dim=1) - 1).clamp(min=0)
    # Extract logits for the last non-padding token of each sequence
    batch_indices = torch.arange(input_batch.size(0), device=device)
    logits = all_logits[batch_indices, last_non_pad_indices, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_classification_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate the average loss over multiple batches from a classification DataLoader.

    Args:
        data_loader: DataLoader providing input and target batches.
        model: GPT model instance.
        device: Device to run computation on.
        num_batches: Number of batches to evaluate. If None, evaluates all batches.

    Returns:
        float: Average loss value.
    """
    model.eval()
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)  # All batches if no number of batches is specified
    else:
        num_batches = min(num_batches, len(data_loader))  # match data loader length if num_batches is higher
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_classification_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches



def evaluate_language_model(model, train_loader, val_loader, device, eval_iter):
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
        train_loss = calc_language_loss_loader(train_loader, model, device, eval_iter)  # Calculate loss on train set
        val_loss = calc_language_loss_loader(val_loader, model, device, eval_iter)  # Calculate loss on validation set
    model.train()
    return train_loss, val_loss

def evaluate_classification_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate the classification model on the train and validation sets.

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
        train_loss = calc_classification_loss_loader(train_loader, model, device, eval_iter)  # Calculate loss on train set
        val_loss = calc_classification_loss_loader(val_loader, model, device, eval_iter)  # Calculate loss on validation set
    model.train()
    return train_loss, val_loss


def train_language_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    train_config: "TrainingConfig",
    tokenizer,
):
    """
    Simple training loop for GPT model.

    Args:
        model: GPT model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer instance.
        device: Device to run training on.
        train_config: Training configuration dictionary.
        tokenizer: Tokenizer object.

    Returns:
        tuple: (train_losses, val_losses, track_tokens_seen)
    """
    # Extract parameters from config
    num_epochs = train_config.get("num_epochs", 1)
    eval_freq = train_config.get("eval_freq", 100)
    eval_iter = train_config.get("eval_iter", 50)
    start_context = train_config.get("start_context", "")
    
    train_losses, val_losses, track_tokens_seen = [], [], []  # Track losses
    tokens_seen, global_step = 0, -1  # Track tokens seen and global step

    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradient from previous batch iteration
            loss = calc_language_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # UPDATE MODEL PARAMETERS
            tokens_seen += input_batch.numel()
            global_step += 1

            # Evaluate model on validation set
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_language_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                logger.info(f"Epoch {epoch+1}, Step {global_step:06d}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Tokens seen: {tokens_seen:08d}")
    
        # Gen and print per Epoch - NOT step
        if start_context:
            generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

def train_classification_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    train_config: "TrainingConfig",
    tokenizer,
):
    """
    Simple training loop for classification model.

    Args:
        model: GPT model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer instance.
        device: Device to run training on.
        train_config: Training configuration dictionary.
        tokenizer: Tokenizer object.

    Returns:
        tuple: (train_losses, val_losses, track_tokens_seen)
    """
    # Extract parameters from config
    num_epochs = train_config.get("num_epochs", 1)
    eval_freq = train_config.get("eval_freq", 100)
    eval_iter = train_config.get("eval_iter", 50)
    start_context = train_config.get("start_context", "")
    
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []  # Track losses
    examples_seen, global_step = 0, -1  # Track examples seen and global step

    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradient from previous batch iteration
            loss = calc_classification_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # UPDATE MODEL PARAMETERS
            examples_seen += input_batch.shape[0]
            global_step += 1

            # Evaluate model on validation set
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_classification_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                logger.info(f"Epoch {epoch+1}, Step {global_step:06d}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}")
    
        # Calculate accuracy after each epoch (use eval_iter batches, not full dataset)
        train_accuracy = calc_classification_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_classification_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        logger.info(f"Train Accuracy: {train_accuracy*100:.2f}%")
        logger.info(f"Validation Accuracy: {val_accuracy*100:.2f}%")
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    return train_losses, val_losses, train_accuracies, val_accuracies, examples_seen