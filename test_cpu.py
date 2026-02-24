"""
Quick CPU test to verify the pipeline works.
Creates synthetic data and runs a minimal training + sampling loop.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def create_synthetic_data(save_path: str, num_seeds: int = 2, num_timesteps: int = 10, size: int = 64):
    """Create small synthetic vorticity-like data for testing."""
    print(f"Creating synthetic data: {num_seeds} seeds, {num_timesteps} timesteps, {size}x{size}")

    data = np.zeros((num_seeds, num_timesteps, size, size), dtype=np.float32)

    for seed in range(num_seeds):
        for t in range(num_timesteps):
            # Create vorticity-like patterns (sinusoidal)
            x = np.linspace(0, 2*np.pi, size)
            y = np.linspace(0, 2*np.pi, size)
            X, Y = np.meshgrid(x, y)

            # Evolving vortex pattern
            phase = t * 0.1 + seed * 0.5
            data[seed, t] = (
                np.sin(4*X + phase) * np.cos(4*Y) +
                0.5 * np.sin(2*X - phase) * np.sin(2*Y + phase) +
                0.1 * np.random.randn(size, size)
            )

    np.save(save_path, data)
    print(f"Saved synthetic data to {save_path}")
    return save_path


def test_imports():
    """Test all imports work."""
    print("Testing imports...")

    from utils import Config, EMAHelper
    from models import UNet, DiffusionSchedule, create_model
    from models.physics import VorticityResidual
    from data import FlowDataset

    print("  All imports OK")
    return True


def test_model():
    """Test model forward pass."""
    print("Testing model...")

    from models import UNet, DiffusionSchedule

    # Small model for testing
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,  # Smaller for CPU
        channel_mult=(1, 1, 2),
        num_res_blocks=1,
        attention_resolutions=(8,),
        dropout=0.0,
        num_groups=8,
        conditional=False,
    )

    # Test forward pass
    x = torch.randn(1, 3, 64, 64)
    t = torch.tensor([500])

    with torch.no_grad():
        out = model(x, t)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    assert out.shape == x.shape, "Output shape mismatch"
    print("  Model forward pass OK")
    return True


def test_diffusion():
    """Test diffusion schedule."""
    print("Testing diffusion schedule...")

    from models import DiffusionSchedule

    schedule = DiffusionSchedule(
        num_timesteps=100,
        beta_schedule="linear",
        device="cpu",
    )

    x = torch.randn(2, 3, 64, 64)
    t = torch.tensor([10, 50])

    x_t, noise = schedule.q_sample(x, t)

    print(f"  Original shape: {x.shape}")
    print(f"  Noised shape: {x_t.shape}")
    print("  Diffusion schedule OK")
    return True


def test_physics():
    """Test physics module."""
    print("Testing physics module...")

    from models.physics import VorticityResidual

    physics = VorticityResidual(
        resolution=64,
        reynolds_number=1000,
        device="cpu",
    )

    # Create test vorticity field (3 timesteps)
    omega = torch.randn(1, 3, 64, 64)

    residual = physics.compute_residual(omega)
    loss = physics.compute_loss(omega)

    print(f"  Input shape: {omega.shape}")
    print(f"  Residual shape: {residual.shape}")
    print(f"  Physics loss: {loss.item():.4f}")
    print("  Physics module OK")
    return True


def test_dataset(data_path: str):
    """Test dataset loading."""
    print("Testing dataset...")

    from data import FlowDataset

    dataset = FlowDataset(
        data_path=data_path,
        train=True,
        train_split=0.8,
        normalize=True,
    )

    sample = dataset[0]
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Sample shape: {sample.shape}")
    print("  Dataset OK")
    return True


def test_training_loop(data_path: str):
    """Test minimal training loop."""
    print("Testing training loop (2 steps)...")

    from models import UNet, DiffusionSchedule
    from data import FlowDataset
    from torch.utils.data import DataLoader

    # Small model
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        channel_mult=(1, 1, 2),
        num_res_blocks=1,
        attention_resolutions=(8,),
        dropout=0.0,
        num_groups=8,
        conditional=False,
    )

    schedule = DiffusionSchedule(num_timesteps=100, device="cpu")

    dataset = FlowDataset(data_path=data_path, train=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = torch.nn.MSELoss()

    model.train()
    for i, batch in enumerate(loader):
        if i >= 2:
            break

        t = torch.randint(0, 100, (batch.shape[0],))
        x_t, noise = schedule.q_sample(batch, t)

        noise_pred = model(x_t, t)
        loss = mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Step {i+1}: loss = {loss.item():.4f}")

    print("  Training loop OK")
    return True


def test_sampling(data_path: str):
    """Test minimal sampling."""
    print("Testing sampling (5 steps)...")

    from models import UNet, DiffusionSchedule

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        channel_mult=(1, 1, 2),
        num_res_blocks=1,
        attention_resolutions=(8,),
        dropout=0.0,
        num_groups=8,
        conditional=False,
    )
    model.eval()

    schedule = DiffusionSchedule(num_timesteps=100, device="cpu")

    # Start from noise
    x = torch.randn(1, 3, 64, 64)

    # Simple sampling loop
    timesteps = [80, 60, 40, 20, 0]

    with torch.no_grad():
        for t in timesteps:
            t_tensor = torch.tensor([t])
            noise_pred = model(x, t_tensor)

            # Simple DDIM-like step
            if t > 0:
                alpha_t = schedule.alphas_cumprod[t]
                alpha_prev = schedule.alphas_cumprod[max(0, t-20)]

                x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * noise_pred

    print(f"  Final output shape: {x.shape}")
    print("  Sampling OK")
    return True


def main():
    print("=" * 50)
    print("CPU Test for diffusion_cfd_clean")
    print("=" * 50)
    print()

    # Create synthetic data
    data_dir = Path(__file__).parent / "data" / "test"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = str(data_dir / "synthetic_test.npy")

    create_synthetic_data(data_path, num_seeds=2, num_timesteps=10, size=64)
    print()

    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Model", test_model),
        ("Diffusion", test_diffusion),
        ("Physics", test_physics),
        ("Dataset", lambda: test_dataset(data_path)),
        ("Training Loop", lambda: test_training_loop(data_path)),
        ("Sampling", lambda: test_sampling(data_path)),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, "FAIL"))
        print()

    # Summary
    print("=" * 50)
    print("Test Results:")
    print("=" * 50)
    for name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {name}: {status}")

    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)
    print()
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\nAll tests passed! Ready for GPU training.")
        return 0
    else:
        print("\nSome tests failed. Please check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
