import argparse
import json
import os
import re
from types import SimpleNamespace

import torch
from tqdm import tqdm
from transformers import (get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup,
                          get_constant_schedule_with_warmup,
                          get_polynomial_decay_schedule_with_warmup,
                          get_inverse_sqrt_schedule)

from gliner import GLiNER
from gliner.modules.base import load_config_as_namespace
from gliner.modules.run_evaluation import get_for_all_path


def save_top_k_checkpoints(model: GLiNER, save_path: str, checkpoint: int, top_k: int = 5):
    """
    Save the top-k checkpoints (latest k checkpoints) of a model and tokenizer.

    Parameters:
        model (GLiNER): The model to save.
        save_path (str): The directory path to save the checkpoints.
        top_k (int): The number of top checkpoints to keep. Defaults to 5.
    """
    # Save the current model and tokenizer
    model.save_pretrained(os.path.join(save_path, checkpoint))
    # tokenizer.save_pretrained(save_path)

    # List all files in the directory
    files = os.listdir(save_path)

    # Filter files to keep only the model checkpoints
    checkpoint_folders = [file for file in files if re.search('model\\_\\d+', file)]

    # Sort checkpoint files by modification time (latest first)
    checkpoint_folders.sort(key=lambda x: os.path.getmtime(os.path.join(save_path, x)), reverse=True)

    # Keep only the top-k checkpoints
    for checkpoint_folder in checkpoint_folders[top_k:]:
        checkpoint_folder = os.path.join(save_path, checkpoint_folder)
        checkpoint_files = [os.path.join(checkpoint_folder, f) for f in os.listdir(checkpoint_folder)]
        for file in checkpoint_files:
            os.remove(file)
        os.rmdir(os.path.join(checkpoint_folder))

# train function
def train(model, optimizer, train_data, num_steps=1000, eval_every=100, log_dir="logs", val_data_dir="none",
          warmup_ratio=0.1, train_batch_size=8, scheduler_type="cosine", save_total_limit = 5, device='cuda'):
    # Set the model to training mode
    model.train()

    # Initialize the training data loader
    train_loader = model.create_dataloader(train_data, batch_size=train_batch_size, shuffle=True)

    # Progress bar setup
    pbar = tqdm(range(num_steps))

    # Compute number of warmup steps for learning rate scheduler
    num_warmup_steps = int(num_steps * warmup_ratio) if warmup_ratio < 1 else int(warmup_ratio)

    # Learning rate scheduler
    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps
        )
    elif scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps
        )
    elif scheduler_type == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
        )
    elif scheduler_type == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps
        )
    elif scheduler_type == "inverse_sqrt":
        scheduler = get_inverse_sqrt_schedule(
            optimizer,
            num_warmup_steps=num_warmup_steps,
        )
    else:
        raise ValueError(
            f"Invalid sheduler_type value: '{scheduler_type} \n Supported sheduler types: 'cosine', 'linear', 'constant', 'polynomial', 'inverse_sqrt'"
        )
    # Create an iterator for the training data loader
    iter_train_loader = iter(train_loader)

    # Gradient scaling to improve training stability
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for step in pbar:
        # Reset gradients
        optimizer.zero_grad()

        # Handle cycling of data loader when running out of data
        try:
            x = next(iter_train_loader)
        except StopIteration:
            iter_train_loader = iter(train_loader)
            x = next(iter_train_loader)

        # Move data to the specified device (e.g., GPU)
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(device)

        # Mixed precision training block
        try:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = model(x)  # Compute loss

            # Check for NaN values in the loss
            if torch.isnan(loss).any():
                print("Warning: NaN loss detected")
                continue

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # Update model parameters
            scaler.update()  # Update scaler for next iteration
            scheduler.step()  # Adjust learning rate
        except Exception as e:
            # Clean up if an error occurs during training
            print(f"Error: {e}")
            torch.cuda.empty_cache()
            continue

        # Update progress bar with current training status
        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
        pbar.set_description(description)

        # Periodically evaluate the model and save a checkpoint
        if (step + 1) % eval_every == 0:
            checkpoint =  f'model_{step + 1}'
            save_top_k_checkpoints(model, log_dir, checkpoint, save_total_limit)
            # Perform validation if a directory is provided
            if val_data_dir != "none":
                get_for_all_path(model, step, log_dir, val_data_dir)

            # Ensure the model is still in training mode after evaluation
            model.train()


def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument('--log_dir', type=str, default='logs', help='Path to the log directory')
    return parser


if __name__ == "__main__":
    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Load configuration from file specified in command-line args
    config = load_config_as_namespace(args.config)
    config.log_dir = args.log_dir  # Override log directory if provided in args

    # Load training data or fall back to generating sample data on failure
    with open(config.train_data, 'r') as f:
        data = json.load(f)

    # Create a model configuration namespace
    model_config = SimpleNamespace(
        # model parameters
        model_name=config.model_name,
        name=config.name,
        max_width=config.max_width,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        fine_tune=config.fine_tune,
        subtoken_pooling=config.subtoken_pooling,
        span_mode=config.span_mode,

        # loss parameters
        loss_alpha=config.loss_alpha,
        loss_gamma=config.loss_gamma,
        loss_reduction=config.loss_reduction,

        # sampling parameters
        max_types=config.max_types,
        shuffle_types=config.shuffle_types,
        random_drop=config.random_drop,
        max_neg_type_ratio=config.max_neg_type_ratio,
        max_len=config.max_len
    )

    if config.prev_path != "none":
        model = GLiNER.from_pretrained(config.prev_path)
        model.config = config
    else:
        model = GLiNER(model_config)

    # Set device to CUDA if available, and move the model to that device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Configure optimizer with specified learning rates and options
    lr_encoder = float(config.lr_encoder)
    lr_others = float(config.lr_others)
    weight_decay_encoder = float(config.weight_decay_encoder)
    weight_decay_others = float(config.weight_decay_other)

    optimizer = model.get_optimizer(lr_encoder, lr_others,
                                    weight_decay_encoder, weight_decay_others,
                                    freeze_token_rep=config.freeze_token_rep)

    # Start the training process with the specified configuration
    train(model, optimizer, data, num_steps=config.num_steps, eval_every=config.eval_every,
          log_dir=config.log_dir, val_data_dir=config.val_data_dir, warmup_ratio=config.warmup_ratio,
          train_batch_size=config.train_batch_size, scheduler_type=config.scheduler_type, 
          save_total_limit=config.save_total_limit,device=device)