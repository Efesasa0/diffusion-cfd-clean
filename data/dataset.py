"""
Dataset classes for CFD flow data.

Handles loading, preprocessing, and batching of Kolmogorov flow data.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict
from pathlib import Path


class FlowDataset(Dataset):
    """
    Dataset for Kolmogorov flow vorticity fields.

    Loads 2D vorticity snapshots and returns triplets of consecutive
    timesteps for temporal context.

    Data format:
        Input NPY/NPZ file should have shape: (num_seeds, num_timesteps, H, W)
        where each seed is an independent simulation.
    """

    def __init__(
        self,
        data_path: str,
        train: bool = True,
        train_split: float = 0.9,
        max_samples: Optional[int] = None,
        normalize: bool = True,
    ):
        """
        Args:
            data_path: Path to .npy or .npz data file
            train: If True, use training split; else test split
            train_split: Fraction of data for training
            max_samples: Maximum samples to load (None = all)
            normalize: Whether to normalize data
        """
        super().__init__()

        self.data_path = Path(data_path)
        self.train = train
        self.normalize = normalize

        # Load data
        self._load_data(max_samples)

        # Split data
        self._split_data(train_split)

        # Compute normalization statistics
        if normalize:
            self._compute_stats()

    def _load_data(self, max_samples: Optional[int]):
        """Load flow data from file."""
        if self.data_path.suffix == '.npy':
            data = np.load(self.data_path)
        elif self.data_path.suffix == '.npz':
            npz = np.load(self.data_path)
            # Assume first array or 'data' key
            key = list(npz.keys())[0] if 'data' not in npz else 'data'
            data = npz[key]
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        # Expected shape: (num_seeds, num_timesteps, H, W)
        if data.ndim == 3:
            # Single seed: (T, H, W) -> (1, T, H, W)
            data = data[None, ...]

        self.num_seeds, self.num_timesteps, self.H, self.W = data.shape

        # Limit samples if specified
        if max_samples is not None:
            num_seeds_to_use = min(max_samples, self.num_seeds)
            data = data[:num_seeds_to_use]
            self.num_seeds = num_seeds_to_use

        self.full_data = data

    def _split_data(self, train_split: float):
        """Split data into train/test sets."""
        total_samples = self.num_seeds * (self.num_timesteps - 2)  # -2 for triplets

        # Create indices for all valid triplets
        all_indices = []
        for seed in range(self.num_seeds):
            for t in range(1, self.num_timesteps - 1):  # Middle timestep of triplet
                all_indices.append((seed, t))

        # Split indices
        n_train = int(len(all_indices) * train_split)

        if self.train:
            self.indices = all_indices[:n_train]
        else:
            self.indices = all_indices[n_train:]

        # Store relevant data
        self.data = self.full_data

    def _compute_stats(self):
        """Compute mean and std for normalization."""
        train_data = self.full_data[:int(self.num_seeds * 0.9)]
        self.mean = train_data.mean()
        self.std = train_data.std()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a triplet of consecutive vorticity fields.

        Returns:
            Tensor of shape (3, H, W) containing [t-1, t, t+1] vorticity
        """
        seed, t = self.indices[idx]

        # Get triplet
        triplet = self.data[seed, t-1:t+2]  # Shape: (3, H, W)

        # Convert to tensor
        triplet = torch.from_numpy(triplet.copy()).float()

        # Normalize
        if self.normalize:
            triplet = (triplet - self.mean) / self.std

        return triplet

    def get_stats(self) -> Dict[str, float]:
        """Return normalization statistics."""
        return {"mean": self.mean, "std": self.std}


class InferenceDataLoader:
    """
    Data loader for inference with low-res and high-res pairs.

    Handles loading sparse/blurred observations and ground truth.
    """

    def __init__(
        self,
        highres_path: str,
        lowres_path: Optional[str] = None,
        lowres_key: str = 'u3232',
        blur_scale: int = 8,
        normalize: bool = True,
    ):
        """
        Args:
            highres_path: Path to high-resolution ground truth
            lowres_path: Path to low-resolution observations (NPZ)
            lowres_key: Key in NPZ file for low-res data
            blur_scale: Downsampling factor for creating low-res
            normalize: Whether to normalize data
        """
        self.blur_scale = blur_scale
        self.normalize = normalize

        # Load high-res data
        self.highres = np.load(highres_path)
        if self.highres.ndim == 3:
            self.highres = self.highres[None, ...]

        # Compute stats from high-res
        self.mean = self.highres.mean()
        self.std = self.highres.std()

        # Load or create low-res data
        if lowres_path is not None:
            npz = np.load(lowres_path)
            self.lowres = npz[lowres_key]
        else:
            # Create low-res by downsampling
            self.lowres = self._create_lowres(self.highres)

    def _create_lowres(self, highres: np.ndarray) -> np.ndarray:
        """Create low-resolution data by downsampling."""
        from scipy.ndimage import zoom

        factor = 1.0 / self.blur_scale
        # Downsample
        lowres = zoom(highres, (1, 1, factor, factor), order=1)
        # Upsample back to original size
        lowres = zoom(lowres, (1, 1, self.blur_scale, self.blur_scale), order=0)
        return lowres

    def get_batch(
        self,
        indices: list,
        timestep: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get batch of low-res and high-res pairs.

        Args:
            indices: Seed indices to load
            timestep: Timestep to load

        Returns:
            Tuple of (lowres, highres) tensors
        """
        lowres_batch = []
        highres_batch = []

        for idx in indices:
            # Get triplet centered at timestep
            hr = self.highres[idx, timestep-1:timestep+2]
            lr = self.lowres[idx, timestep-1:timestep+2]

            if self.normalize:
                hr = (hr - self.mean) / self.std
                lr = (lr - self.mean) / self.std

            highres_batch.append(hr)
            lowres_batch.append(lr)

        lowres = torch.from_numpy(np.stack(lowres_batch)).float()
        highres = torch.from_numpy(np.stack(highres_batch)).float()

        return lowres, highres

    def get_stats(self) -> Dict[str, float]:
        """Return normalization statistics."""
        return {"mean": self.mean, "std": self.std}


def create_dataloader(
    config,
    train: bool = True,
) -> DataLoader:
    """Create DataLoader from config."""
    dataset = FlowDataset(
        data_path=config.data.data_dir,
        train=train,
        train_split=config.data.train_split,
        max_samples=config.data.max_samples,
        normalize=True,
    )

    return DataLoader(
        dataset,
        batch_size=config.training.batch_size if train else config.sampling.batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True,
        drop_last=train,
    )
