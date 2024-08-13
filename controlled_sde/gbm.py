"""Provides the non-linear attractor SDE."""
import torch
from .controlled_sde import ControlledSDE


class GBMDrift(torch.nn.Module):
    """
    Drift module for the attractor.
    """

    def forward(self, input: torch.Tensor, u: torch.Tensor):
        """forward function of the attractor drift module"""
        # split the input x into X and Y
        x, y = torch.split(input, split_size_or_sections=(1, 1), dim=1)
        # compute the drift components
        f1 = -0.5 * x + y
        f2 = -x - 0.5 * y
        # combine and return
        return torch.cat([f1, f2], dim=1) + u


class GBMDiffusion(torch.nn.Module):
    """
    Diffusion module for the attractor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, _u: torch.Tensor):
        """forward function of the attractor diffusion module"""
        return 0.2 * input  # torch.cat([g1, g2], dim=1)


class GBMPolicy(torch.nn.Module):
    """
    Policy module for the attractor.
    """

    def forward(self, input: torch.Tensor):
        """forward function of the attractor policy module"""
        return -input


class GBM(ControlledSDE):
    """
    Geometric Brownian motion problem from the pape
    """

    def __init__(self):
        # initialize the drift and diffusion modules
        policy = GBMPolicy()
        drift = GBMDrift()
        diffusion = GBMDiffusion()
        super().__init__(policy, drift, diffusion, "diagonal", "ito")

    def n_dimensions(self) -> int:
        return 2
