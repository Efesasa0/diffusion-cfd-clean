"""Exponential Moving Average for model weights."""

import torch
import torch.nn as nn
from typing import Optional


class EMAHelper:
    """
    Exponential Moving Average helper for model parameters.

    Maintains a shadow copy of model weights that are updated with:
        shadow = decay * shadow + (1 - decay) * current
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register(model)

    def register(self, model: nn.Module) -> None:
        """Register model parameters for EMA tracking."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """Update shadow weights with current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Apply shadow weights to model (backup current weights first)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module) -> None:
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> dict:
        """Return EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]
