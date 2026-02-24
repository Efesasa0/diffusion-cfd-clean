"""
Training script for physics-informed diffusion model.

Usage:
    python train.py --config configs/default.yaml --mode train
"""

import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from utils import Config, get_arg_parser, EMAHelper
from models import UNet, create_model, DiffusionSchedule
from models.physics import VorticityResidual, create_physics_module
from data import create_dataloader


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config: Config, device: str, checkpoint_path: str = None):
    """
    Main training loop.

    Args:
        config: Configuration object
        device: Device to train on
        checkpoint_path: Optional path to resume from
    """
    # Create directories
    Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize tensorboard
    writer = SummaryWriter(config.paths.log_dir)

    # Create model
    model = create_model(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create diffusion schedule
    schedule = DiffusionSchedule(
        num_timesteps=config.diffusion.num_timesteps,
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        device=device,
    )

    # Create physics module (for conditional training)
    physics = None
    if config.model.type == "conditional":
        physics = create_physics_module(config, device)

    # Create dataloader
    train_loader = create_dataloader(config, train=True)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # EMA
    ema = None
    if config.training.use_ema:
        ema = EMAHelper(model, decay=config.training.ema_rate)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
        if ema is not None and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # Loss function
    mse_loss = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(start_epoch, config.training.num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}")
        for batch in pbar:
            batch = batch.to(device)
            batch_size = batch.shape[0]

            # Sample random timesteps
            t = torch.randint(
                0, config.diffusion.num_timesteps,
                (batch_size,), device=device
            )

            # Add noise
            x_t, noise = schedule.q_sample(batch, t)

            # Compute physics gradient for conditional model
            cond = None
            if config.model.type == "conditional" and physics is not None:
                # 10% of time use physics conditioning
                if torch.rand(1).item() < 0.1:
                    with torch.enable_grad():
                        cond = physics.compute_gradient(batch)

            # Forward pass
            noise_pred = model(x_t, t, cond)

            # Compute loss
            loss = mse_loss(noise_pred, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if config.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

            optimizer.step()

            # EMA update
            if ema is not None:
                ema.update(model)

            # Logging
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            pbar.set_postfix({"loss": loss.item()})

            if global_step % config.training.log_freq == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)

            # Save checkpoint
            if global_step % config.training.save_freq == 0:
                save_checkpoint(
                    model, optimizer, ema, epoch, global_step,
                    config.paths.checkpoint_dir
                )

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

    # Final checkpoint
    save_checkpoint(
        model, optimizer, ema, config.training.num_epochs, global_step,
        config.paths.checkpoint_dir, final=True
    )

    writer.close()
    print("Training complete!")


def save_checkpoint(
    model, optimizer, ema, epoch, global_step, save_dir, final=False
):
    """Save training checkpoint."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()

    if final:
        path = Path(save_dir) / "final_model.pth"
    else:
        path = Path(save_dir) / f"checkpoint_step{global_step}.pth"

    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)

    # Set seed
    set_seed(args.seed)

    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Run training
    train(config, device, args.checkpoint)
