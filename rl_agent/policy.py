"""Provides the policy used in the experiments."""

from torch import nn


class TanhPolicy(nn.Sequential):
    """
    A policy with three layers and tanh activations.
    """

    def __init__(self, n_out: int = 1, n_hidden: int = 64):
        super().__init__(
            nn.LazyLinear(n_hidden),
            nn.Tanh(),
            nn.LazyLinear(n_hidden),
            nn.Tanh(),
            nn.LazyLinear(n_out),
        )
