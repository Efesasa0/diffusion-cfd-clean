"""Configuration utilities for loading and managing YAML configs."""

import yaml
import argparse
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


@dataclass
class DataConfig:
    dataset: str = "kolmogorov_flow"
    data_dir: str = "./data"
    image_size: int = 256
    channels: int = 3
    train_split: float = 0.9
    max_samples: Optional[int] = None


@dataclass
class ModelConfig:
    type: str = "conditional"
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 64
    channel_mult: tuple = (1, 1, 1, 2)
    num_res_blocks: int = 1
    attention_resolutions: tuple = (16,)
    dropout: float = 0.1
    num_groups: int = 8


@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02


@dataclass
class TrainingConfig:
    batch_size: int = 2
    num_epochs: int = 100
    learning_rate: float = 0.0002
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_rate: float = 0.9999
    use_ema: bool = True
    save_freq: int = 10000
    log_freq: int = 100


@dataclass
class SamplingConfig:
    batch_size: int = 4
    num_steps: int = 50
    eta: float = 0.0
    guidance_scale: float = 0.0
    physics_weight: float = 0.0


@dataclass
class PhysicsConfig:
    reynolds_number: float = 1000
    dt: float = 0.03125
    domain_size: float = 6.283185307179586


@dataclass
class PathConfig:
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    output_dir: str = "./outputs"


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**{
                k: tuple(v) if isinstance(v, list) else v
                for k, v in config_dict.get("model", {}).items()
            }),
            diffusion=DiffusionConfig(**config_dict.get("diffusion", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            sampling=SamplingConfig(**config_dict.get("sampling", {})),
            physics=PhysicsConfig(**config_dict.get("physics", {})),
            paths=PathConfig(**config_dict.get("paths", {})),
        )

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load Config from YAML file."""
        config_dict = load_config(config_path)
        return cls.from_dict(config_dict)


def get_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(description="Physics-Informed Diffusion for CFD")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["train", "sample"], required=True,
                        help="Run mode: train or sample")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for resuming/sampling")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser
