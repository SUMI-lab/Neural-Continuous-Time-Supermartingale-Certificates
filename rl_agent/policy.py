"""Provides the policy used in the experiments."""
import torch
from torch import nn


class TanhPolicy(nn.Sequential):
    """
    A policy with three layers and tanh activations.
    """

    def __init__(self, n_out: int = 1, n_hidden: int = 64, device: torch.device | str = "cpu"):
        super().__init__(
            nn.LazyLinear(n_hidden, dtype=torch.float32, device=device),
            nn.Tanh(),
            nn.LazyLinear(n_hidden, dtype=torch.float32, device=device),
            nn.Tanh(),
            nn.LazyLinear(n_out, dtype=torch.float32, device=device),
        )
