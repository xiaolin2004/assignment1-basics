import time
import os
import numpy as np
import numpy.typing as npt
import torch
import typing
import logging

from cs336_basics.assignment1.data_loader import data_loader
from cs336_basics.assignment1.checkpoint import save_checkpoint
from cs336_basics.assignment1.cross_entrpy import cross_entropy

def estimate_loss(
    model: torch.nn.Module,
    data: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    """Estimate loss on a given dataset split."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = data_loader(data, batch_size, context_length, device)
        with torch.no_grad():
            logits = model(X)
            # Reshape for cross_entropy: (B*T, V) and (B*T)
            # Our custom cross_entropy implementation expects (B, V) and (B) or handled internally?
            # Looking at previous implementations:
            # run_transformer_lm returns (B, T, V)
            # run_cross_entropy expects (B, V) and (B) which implies per-token loss averaging.
            # But we are dealing with sequences.
            # The standard way is to flatten:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            Y = Y.view(B * T)
            loss = cross_entropy(logits, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: npt.NDArray,
    val_data: npt.NDArray,
    num_iters: int,
    batch_size: int,
    context_length: int,
    device: str,
    log_interval: int = 100,
    eval_interval: int = 500,
    eval_iters: int = 200,
    save_interval: int = 1000,
    checkpoint_pat_prefix: str = "checkpoints/model",
    wandb_log: bool = False,
):
    model.to(device)
    model.train()
    
    # Initialize iteration counter
    start_iter = 0 
    # (In a real scenario we might load 'start_iter' from a checkpoint if resuming)

    t0 = time.time()
    for iter_num in range(start_iter, num_iters + 1):
        
        # 1. Fetch batch
        X, Y = data_loader(train_data, batch_size, context_length, device)
        
        # 2. Forward pass
        logits = model(X)
        
        # 3. Calculate loss
        B, T, V = logits.shape
        logits_reshaped = logits.view(B * T, V)
        targets_reshaped = Y.view(B * T)
        loss = cross_entropy(logits_reshaped, targets_reshaped)
        
        # 4. Backward pass
        loss.backward()
        
        # 5. Optimizer step (Gradient Clipping is typically done before step, but not strictly requested here yet, 
        #    but previous helper had it. We can add it if we have the helper import available, 
        #    but we'll stick to basic optimizer step for now unless requested).
        optimizer.step()
        optimizer.zero_grad()
        
        # Logging
        if iter_num % log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            # Basic console log
            print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")
            if wandb_log:
                import wandb
                wandb.log({
                    "iter": iter_num,
                    "train/loss": loss.item(),
                    "train/time_ms": dt * 1000,
                    "lr": optimizer.param_groups[0]['lr']
                })
        
        # Evaluation
        if iter_num > 0 and iter_num % eval_interval == 0:
            val_loss = estimate_loss(model, val_data, batch_size, context_length, device, eval_iters)
            print(f"step {iter_num}: val loss {val_loss:.4f}")
            if wandb_log:
                import wandb
                wandb.log({
                    "iter": iter_num,
                    "val/loss": val_loss,
                })

        # Checkpointing
        if iter_num > 0 and iter_num % save_interval == 0:
            checkpoint_path = f"{checkpoint_pat_prefix}_{iter_num}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            print(f"Saving checkpoint to {checkpoint_path}")
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)

