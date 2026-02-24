"""
Physics-informed components for CFD diffusion models.

Implements vorticity residual computation using spectral methods (FFT)
for the 2D Navier-Stokes equations (Kolmogorov flow).
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class VorticityResidual:
    """
    Computes vorticity transport equation residual using spectral methods.

    The 2D vorticity transport equation is:
        ∂ω/∂t + u·∇ω = ν∇²ω + f

    where:
        - ω is vorticity
        - u = (u, v) is velocity field
        - ν = 1/Re is kinematic viscosity
        - f is forcing term (Kolmogorov forcing: -4cos(4y))

    Uses FFT-based spectral derivatives for accuracy with periodic boundaries.
    """

    def __init__(
        self,
        resolution: int = 256,
        reynolds_number: float = 1000.0,
        dt: float = 0.03125,
        domain_size: float = 2 * math.pi,
        device: str = "cuda",
    ):
        self.resolution = resolution
        self.Re = reynolds_number
        self.nu = 1.0 / reynolds_number
        self.dt = dt
        self.L = domain_size
        self.device = device

        # Precompute wavenumbers
        self._setup_spectral_operators()

    def _setup_spectral_operators(self):
        """Setup wavenumber arrays for spectral derivatives."""
        n = self.resolution
        # Wavenumbers: 0, 1, 2, ..., n/2-1, -n/2, -n/2+1, ..., -1
        k = torch.fft.fftfreq(n, d=1/n).to(self.device) * 2 * math.pi / self.L

        # 2D wavenumber grids
        self.kx = k[None, :].expand(n, n)
        self.ky = k[:, None].expand(n, n)

        # Laplacian in spectral space: -(kx² + ky²)
        self.k_squared = self.kx**2 + self.ky**2
        # Avoid division by zero at k=0
        self.k_squared_inv = torch.where(
            self.k_squared == 0,
            torch.zeros_like(self.k_squared),
            1.0 / self.k_squared
        )

        # Physical space coordinates for forcing
        y = torch.linspace(0, self.L, n, device=self.device)
        self.forcing = -4.0 * torch.cos(4.0 * y)[:, None].expand(n, n)

    def spectral_derivative(
        self,
        field: torch.Tensor,
        direction: str,
    ) -> torch.Tensor:
        """
        Compute derivative using FFT.

        Args:
            field: Input field (B, H, W) or (B, C, H, W)
            direction: 'x' or 'y'

        Returns:
            Derivative field
        """
        squeeze = False
        if field.dim() == 3:
            field = field.unsqueeze(1)
            squeeze = True

        B, C, H, W = field.shape
        kx, ky, _ = self._get_spectral_operators(H, W, field.device)

        # FFT
        field_hat = torch.fft.fft2(field)

        # Multiply by ik
        if direction == 'x':
            deriv_hat = 1j * kx * field_hat
        else:  # y
            deriv_hat = 1j * ky * field_hat

        # Inverse FFT
        deriv = torch.fft.ifft2(deriv_hat).real

        if squeeze:
            deriv = deriv.squeeze(1)

        return deriv

    def _get_k_squared(self, H: int, W: int, device: torch.device):
        """Get k_squared for given resolution."""
        if H == self.resolution and W == self.resolution:
            return self.k_squared

        k_y = torch.fft.fftfreq(H, d=1/H, device=device) * 2 * math.pi / self.L
        k_x = torch.fft.fftfreq(W, d=1/W, device=device) * 2 * math.pi / self.L
        kx = k_x[None, :].expand(H, W)
        ky = k_y[:, None].expand(H, W)
        return kx**2 + ky**2

    def _get_forcing(self, H: int, W: int, device: torch.device):
        """Get forcing term for given resolution."""
        if H == self.resolution and W == self.resolution:
            return self.forcing

        y = torch.linspace(0, self.L, H, device=device)
        return -4.0 * torch.cos(4.0 * y)[:, None].expand(H, W)

    def laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian using FFT."""
        squeeze = False
        if field.dim() == 3:
            field = field.unsqueeze(1)
            squeeze = True

        B, C, H, W = field.shape
        k_squared = self._get_k_squared(H, W, field.device)

        field_hat = torch.fft.fft2(field)
        lap_hat = -k_squared * field_hat
        lap = torch.fft.ifft2(lap_hat).real

        if squeeze:
            lap = lap.squeeze(1)

        return lap

    def _get_spectral_operators(self, H: int, W: int, device: torch.device):
        """Get or compute spectral operators for given resolution."""
        if H == self.resolution and W == self.resolution:
            return self.kx, self.ky, self.k_squared_inv

        # Dynamically compute for different resolution
        k_y = torch.fft.fftfreq(H, d=1/H, device=device) * 2 * math.pi / self.L
        k_x = torch.fft.fftfreq(W, d=1/W, device=device) * 2 * math.pi / self.L
        kx = k_x[None, :].expand(H, W)
        ky = k_y[:, None].expand(H, W)
        k_squared = kx**2 + ky**2
        k_squared_inv = torch.where(
            k_squared == 0,
            torch.zeros_like(k_squared),
            1.0 / k_squared
        )
        return kx, ky, k_squared_inv

    def vorticity_to_velocity(
        self,
        omega: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute velocity from vorticity using stream function.

        ω = ∂v/∂x - ∂u/∂y
        ∇²ψ = ω
        u = ∂ψ/∂y, v = -∂ψ/∂x
        """
        squeeze = False
        if omega.dim() == 3:
            omega = omega.unsqueeze(1)
            squeeze = True

        B, C, H, W = omega.shape
        kx, ky, k_squared_inv = self._get_spectral_operators(H, W, omega.device)

        # Solve Poisson equation for stream function
        omega_hat = torch.fft.fft2(omega)
        psi_hat = -omega_hat * k_squared_inv

        # Compute velocity from stream function
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat

        u = torch.fft.ifft2(u_hat).real
        v = torch.fft.ifft2(v_hat).real

        if squeeze:
            u = u.squeeze(1)
            v = v.squeeze(1)

        return u, v

    def compute_residual(
        self,
        omega: torch.Tensor,
        omega_prev: Optional[torch.Tensor] = None,
        omega_next: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute vorticity transport equation residual.

        Args:
            omega: Current vorticity field (B, H, W) or (B, 3, H, W) for 3 timesteps
            omega_prev: Previous timestep (optional, uses omega[:, 0] if omega has 3 channels)
            omega_next: Next timestep (optional, uses omega[:, 2] if omega has 3 channels)

        Returns:
            Residual field
        """
        # Handle 3-channel input (3 consecutive timesteps)
        if omega.dim() == 4 and omega.shape[1] == 3:
            omega_prev = omega[:, 0]
            omega_curr = omega[:, 1]
            omega_next = omega[:, 2]
        else:
            omega_curr = omega

        # Compute velocity from vorticity
        u, v = self.vorticity_to_velocity(omega_curr)

        # Compute spatial derivatives
        domega_dx = self.spectral_derivative(omega_curr, 'x')
        domega_dy = self.spectral_derivative(omega_curr, 'y')

        # Advection term: u · ∇ω
        advection = u * domega_dx + v * domega_dy

        # Diffusion term: ν∇²ω
        diffusion = self.nu * self.laplacian(omega_curr)

        # Time derivative (if we have temporal data)
        if omega_prev is not None and omega_next is not None:
            domega_dt = (omega_next - omega_prev) / (2 * self.dt)
        elif omega_next is not None:
            domega_dt = (omega_next - omega_curr) / self.dt
        elif omega_prev is not None:
            domega_dt = (omega_curr - omega_prev) / self.dt
        else:
            domega_dt = torch.zeros_like(omega_curr)

        # Forcing term (Kolmogorov flow) - get for current resolution
        H, W = omega_curr.shape[-2:]
        forcing = self._get_forcing(H, W, omega_curr.device)

        # Residual: ∂ω/∂t + u·∇ω - ν∇²ω - f
        # With linear damping term (0.1ω) as in original code
        residual = domega_dt + advection - diffusion + 0.1 * omega_curr - forcing

        return residual

    def compute_loss(
        self,
        omega: torch.Tensor,
        omega_prev: Optional[torch.Tensor] = None,
        omega_next: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mean squared residual loss."""
        residual = self.compute_residual(omega, omega_prev, omega_next)
        return torch.mean(residual ** 2)

    def compute_gradient(
        self,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient of physics loss with respect to omega.

        This is used for physics-informed guidance during sampling.
        """
        omega = omega.detach().requires_grad_(True)
        loss = self.compute_loss(omega)
        grad = torch.autograd.grad(loss, omega)[0]
        return grad


def create_physics_module(config, device: str = "cuda") -> VorticityResidual:
    """Create physics module from config."""
    return VorticityResidual(
        resolution=config.data.image_size,
        reynolds_number=config.physics.reynolds_number,
        dt=config.physics.dt,
        domain_size=config.physics.domain_size,
        device=device,
    )
