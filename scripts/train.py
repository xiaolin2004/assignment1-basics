
import argparse
import os
import random
import numpy as np
import torch

from cs336_basics.assignment1.transformer import Transformer_lm
from cs336_basics.assignment1.training import train
from cs336_basics.assignment1.adam_w import AdamW

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")
    
    # Data params
    parser.add_argument("--input_path", type=str, required=True, help="Path to input .npy file (e.g. data/train.npy)")
    parser.add_argument("--val_path", type=str, default=None, help="Path to validation .npy file. If None, split form input_path not implemented in this simple script yet, so better provide it.")
    
    # Model params
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    
    # Training params
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    
    # Logging and Checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default="transformer-run", help="WandB run name")
    
    # System
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device
    device = args.device
    print(f"Using device: {device}")

    # Load data using memory mapping
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    
    train_data = np.memmap(args.input_path, dtype=np.uint16, mode='r')
    if args.val_path and os.path.exists(args.val_path):
        val_data = np.memmap(args.val_path, dtype=np.uint16, mode='r')
    else:
        # Fallback: simple split 90/10 if no val path provided? 
        # For simplicity, if no val_path, just use same data (not ideal but avoids crash)
        # Or better: raise error if strictly needed. Let's assume user provides it or we use train_data for both (bad practice but works for code check)
        print("Warning: No validation path provided. Using training data for validation.")
        val_data = train_data

    # Initialize Model
    model = Transformer_lm(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        max_seq_len=args.context_length,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        device=device,
    )

    # Initialize Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    # Setup WandB
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        except ImportError:
            print("WandB is not installed. Continuing without WandB logging.")
            args.wandb = False

    # Start Training
    try:
        train(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            val_data=val_data,
            num_iters=args.num_iters,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            eval_iters=100, # Hardcoded for now, or could be arg
            save_interval=args.save_interval,
            checkpoint_pat_prefix=os.path.join(args.checkpoint_dir, "model"),
            wandb_log=args.wandb,
            log_dir=args.checkpoint_dir,
        )
    except KeyboardInterrupt:
        print("Training broken by user...")

if __name__ == "__main__":
    main()
