"""Diffusion process utilities: beta schedules, noise, sampling."""

import math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def get_beta_schedule(
    schedule: str,
    num_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """
    Create beta schedule for diffusion process.

    Args:
        schedule: Type of schedule ("linear", "cosine", "quadratic")
        num_timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value

    Returns:
        Beta values tensor of shape (num_timesteps,)
    """
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
    elif schedule == "quadratic":
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
    elif schedule == "cosine":
        # Cosine schedule from "Improved DDPM" paper
        steps = num_timesteps + 1
        s = 0.008
        t = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos((t / num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clamp(betas, min=0.0001, max=0.9999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")

    return betas


class DiffusionSchedule:
    """
    Precomputed diffusion schedule values.

    Stores alpha, alpha_cumprod, and other derived quantities
    needed for training and sampling.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda",
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        # Beta schedule
        betas = get_beta_schedule(beta_schedule, num_timesteps, beta_start, beta_end)
        self.betas = betas.to(device)

        # Alpha values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Derived quantities for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Derived quantities for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from q(x_t | x_0) - the forward diffusion process.

        Args:
            x_0: Clean data of shape (B, C, H, W)
            t: Timesteps of shape (B,)
            noise: Optional pre-generated noise

        Returns:
            Tuple of (noisy_x, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return (x_t - sqrt_one_minus_alpha * noise) / sqrt_alpha

    def q_posterior_mean(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean of posterior q(x_{t-1} | x_t, x_0)."""
        coef1 = self.posterior_mean_coef1[t][:, None, None, None]
        coef2 = self.posterior_mean_coef2[t][:, None, None, None]
        return coef1 * x_0 + coef2 * x_t


def ddpm_sample_step(
    model_output: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    schedule: DiffusionSchedule,
    eta: float = 1.0,
) -> torch.Tensor:
    """
    Single DDPM/DDIM sampling step.

    Args:
        model_output: Predicted noise from model
        x_t: Current noisy sample
        t: Current timestep
        schedule: Diffusion schedule
        eta: Noise scale (0 = DDIM, 1 = DDPM)

    Returns:
        x_{t-1}: Denoised sample
    """
    batch_size = x_t.shape[0]
    device = x_t.device
    t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

    # Predict x_0
    x_0_pred = schedule.predict_x0_from_noise(x_t, t_tensor, model_output)
    x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

    if t == 0:
        return x_0_pred

    # Compute variance
    alpha_t = schedule.alphas_cumprod[t]
    alpha_t_prev = schedule.alphas_cumprod[t - 1]

    # DDIM formula
    sigma = eta * torch.sqrt(
        (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
    )

    # Direction pointing to x_t
    pred_dir = torch.sqrt(1 - alpha_t_prev - sigma**2) * model_output

    # Combine
    x_prev = torch.sqrt(alpha_t_prev) * x_0_pred + pred_dir

    if eta > 0:
        noise = torch.randn_like(x_t)
        x_prev = x_prev + sigma * noise

    return x_prev


@torch.no_grad()
def ddim_sample(
    model,
    schedule: DiffusionSchedule,
    shape: Tuple[int, ...],
    num_steps: int = 50,
    eta: float = 0.0,
    device: str = "cuda",
    guidance_func=None,
    guidance_scale: float = 0.0,
) -> torch.Tensor:
    """
    Full DDIM sampling loop.

    Args:
        model: Denoising model
        schedule: Diffusion schedule
        shape: Output shape (B, C, H, W)
        num_steps: Number of sampling steps
        eta: Noise scale (0 = deterministic DDIM)
        device: Device to use
        guidance_func: Optional guidance function
        guidance_scale: Scale for guidance

    Returns:
        Generated samples
    """
    # Timestep sequence (evenly spaced)
    step_size = schedule.num_timesteps // num_steps
    timesteps = list(range(0, schedule.num_timesteps, step_size))[::-1]

    # Start from pure noise
    x = torch.randn(shape, device=device)

    for t in timesteps:
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Get model prediction
        noise_pred = model(x, t_tensor)

        # Apply guidance if provided
        if guidance_func is not None and guidance_scale > 0:
            with torch.enable_grad():
                x_grad = x.detach().requires_grad_(True)
                guidance = guidance_func(x_grad)
                grad = torch.autograd.grad(guidance.sum(), x_grad)[0]
            noise_pred = noise_pred - guidance_scale * grad

        # Take DDIM step
        x = ddpm_sample_step(noise_pred, x, t, schedule, eta)

    return x
