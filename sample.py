"""
Sampling/inference script for physics-informed diffusion model.

Reconstructs high-resolution flow fields from low-resolution observations
using guided diffusion sampling.

Usage:
    python sample.py --config configs/default.yaml --mode sample --checkpoint checkpoints/final_model.pth
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from utils import Config, get_arg_parser
from models import UNet, create_model, DiffusionSchedule
from models.physics import create_physics_module
from data import InferenceDataLoader


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@torch.no_grad()
def guided_ddim_sample(
    model: UNet,
    schedule: DiffusionSchedule,
    x_init: torch.Tensor,
    start_timestep: int,
    num_steps: int = 50,
    eta: float = 0.0,
    physics_module=None,
    physics_weight: float = 0.0,
    guidance_scale: float = 0.0,
) -> torch.Tensor:
    """
    Guided DDIM sampling starting from partially noised input.

    Args:
        model: Denoising model
        schedule: Diffusion schedule
        x_init: Initial (potentially low-res) input
        start_timestep: Starting diffusion timestep
        num_steps: Number of denoising steps
        eta: DDIM stochasticity (0 = deterministic)
        physics_module: Optional physics module for guidance
        physics_weight: Weight for physics guidance
        guidance_scale: Classifier-free guidance scale

    Returns:
        Reconstructed high-resolution output
    """
    device = x_init.device
    batch_size = x_init.shape[0]

    # Add noise to starting point
    t_start = torch.full((batch_size,), start_timestep, device=device, dtype=torch.long)
    noise = torch.randn_like(x_init)
    x = schedule.q_sample(x_init, t_start, noise)[0]

    # Create timestep sequence
    step_size = max(1, start_timestep // num_steps)
    timesteps = list(range(start_timestep, 0, -step_size))

    for t in tqdm(timesteps, desc="Sampling", leave=False):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Get physics gradient for conditioning
        cond = None
        if physics_module is not None and model.conditional:
            cond = physics_module.compute_gradient(x.detach())

        # Model prediction
        noise_pred = model(x, t_tensor, cond)

        # Physics guidance (gradient-based)
        if physics_module is not None and physics_weight > 0:
            with torch.enable_grad():
                x_grad = x.detach().requires_grad_(True)
                physics_loss = physics_module.compute_loss(x_grad)
                grad = torch.autograd.grad(physics_loss, x_grad)[0]
            noise_pred = noise_pred + physics_weight * grad

        # DDIM update
        x = ddim_step(x, noise_pred, t, schedule, eta)

    return x


def ddim_step(
    x: torch.Tensor,
    noise_pred: torch.Tensor,
    t: int,
    schedule: DiffusionSchedule,
    eta: float = 0.0,
) -> torch.Tensor:
    """Single DDIM denoising step."""
    batch_size = x.shape[0]
    device = x.device

    t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

    # Predict x_0
    alpha_t = schedule.alphas_cumprod[t]
    sqrt_alpha = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_t)

    x_0_pred = (x - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
    x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

    if t <= 1:
        return x_0_pred

    # Get previous alpha
    t_prev = max(0, t - 1)
    alpha_t_prev = schedule.alphas_cumprod[t_prev]

    # DDIM formula
    sigma = eta * torch.sqrt(
        (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
    )

    pred_dir = torch.sqrt(1 - alpha_t_prev - sigma**2) * noise_pred
    x_prev = torch.sqrt(alpha_t_prev) * x_0_pred + pred_dir

    if eta > 0:
        noise = torch.randn_like(x)
        x_prev = x_prev + sigma * noise

    return x_prev


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute reconstruction metrics."""
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - target))

    # Relative L2 error
    rel_l2 = np.linalg.norm(pred - target) / np.linalg.norm(target)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "rel_l2": rel_l2,
    }


def visualize_results(
    lowres: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    save_path: str,
    sample_idx: int = 0,
):
    """Create visualization of reconstruction results."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Use middle timestep of triplet
    channel = 1

    vmin = min(lowres[sample_idx, channel].min(), target[sample_idx, channel].min())
    vmax = max(lowres[sample_idx, channel].max(), target[sample_idx, channel].max())

    # Low-res input
    im = axes[0].imshow(lowres[sample_idx, channel], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title("Low-res Input")
    axes[0].axis('off')

    # Prediction
    axes[1].imshow(pred[sample_idx, channel], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')

    # Ground truth
    axes[2].imshow(target[sample_idx, channel], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[2].set_title("Ground Truth")
    axes[2].axis('off')

    # Error
    error = np.abs(pred[sample_idx, channel] - target[sample_idx, channel])
    axes[3].imshow(error, cmap='hot')
    axes[3].set_title(f"Absolute Error (max={error.max():.3f})")
    axes[3].axis('off')

    plt.colorbar(im, ax=axes[:3], shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def sample(config: Config, device: str, checkpoint_path: str):
    """
    Run sampling/inference.

    Args:
        config: Configuration object
        device: Device to use
        checkpoint_path: Path to model checkpoint
    """
    # Create output directory
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = create_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    # Create diffusion schedule
    schedule = DiffusionSchedule(
        num_timesteps=config.diffusion.num_timesteps,
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        device=device,
    )

    # Create physics module
    physics = None
    if config.model.type == "conditional" or config.sampling.physics_weight > 0:
        physics = create_physics_module(config, device)

    # Load data
    # Note: Update these paths based on your data location
    data_loader = InferenceDataLoader(
        highres_path=config.data.data_dir,
        normalize=True,
    )

    # Sampling parameters
    batch_size = config.sampling.batch_size
    num_steps = config.sampling.num_steps
    eta = config.sampling.eta
    physics_weight = config.sampling.physics_weight

    # Run reconstruction
    all_predictions = []
    all_targets = []
    all_inputs = []
    all_metrics = []

    # Sample a few test cases
    num_samples = min(10, data_loader.highres.shape[0])
    timestep = 100  # Middle of the simulation

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        indices = list(range(batch_start, batch_end))

        # Get batch
        lowres, highres = data_loader.get_batch(indices, timestep)
        lowres = lowres.to(device)
        highres = highres.to(device)

        # Run guided sampling
        # Start from moderate noise level
        start_t = int(config.diffusion.num_timesteps * 0.3)

        pred = guided_ddim_sample(
            model=model,
            schedule=schedule,
            x_init=lowres,
            start_timestep=start_t,
            num_steps=num_steps,
            eta=eta,
            physics_module=physics,
            physics_weight=physics_weight,
        )

        # Unnormalize
        stats = data_loader.get_stats()
        pred_np = pred.cpu().numpy() * stats["std"] + stats["mean"]
        highres_np = highres.cpu().numpy() * stats["std"] + stats["mean"]
        lowres_np = lowres.cpu().numpy() * stats["std"] + stats["mean"]

        all_predictions.append(pred_np)
        all_targets.append(highres_np)
        all_inputs.append(lowres_np)

        # Compute metrics
        for i in range(pred_np.shape[0]):
            metrics = compute_metrics(pred_np[i], highres_np[i])
            all_metrics.append(metrics)
            print(f"Sample {batch_start + i}: RMSE={metrics['rmse']:.4f}, RelL2={metrics['rel_l2']:.4f}")

    # Aggregate results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_inputs = np.concatenate(all_inputs, axis=0)

    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    print(f"\nAverage Metrics:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.6f}")

    # Save results
    np.savez(
        output_dir / "results.npz",
        predictions=all_predictions,
        targets=all_targets,
        inputs=all_inputs,
    )

    # Visualize first few samples
    for i in range(min(5, len(all_predictions))):
        visualize_results(
            all_inputs, all_predictions, all_targets,
            output_dir / f"visualization_{i}.png",
            sample_idx=i,
        )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for sampling")

    # Load config
    config = Config.from_yaml(args.config)

    # Set seed
    set_seed(args.seed)

    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Run sampling
    sample(config, device, args.checkpoint)
