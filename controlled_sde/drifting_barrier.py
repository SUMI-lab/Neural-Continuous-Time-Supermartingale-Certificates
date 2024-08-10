"""Provides the drifting barrier SDE."""
import torch
from .controlled_sde import ControlledSDE


class BarrierDrift(torch.nn.Module):
    """
    Drift module for the illustrative example from the paper.
    """

    def __init__(self, drift):
        super().__init__()
        self.drift = drift

    def forward(self, x: torch.Tensor, _u: torch.Tensor):
        """forward function of the illustrative example drift module"""
        return self.drift * x


class BarrierDiffusion(torch.nn.Module):
    """
    Diffusion module for the illustrative example from the paper.
    """

    def forward(self, _x: torch.Tensor, u: torch.Tensor):
        """forward function of the illustrative example diffusion module"""
        return u


class DriftingBarrier(ControlledSDE):
    """
    Stochastic drifting barrier SDE from the paper's illustrative example.
    """

    def __init__(self, policy: torch.nn.Module, drift: float = 0.4):
        drift_module = BarrierDrift(drift)
        diffusion_module = BarrierDiffusion()
        super().__init__(
            policy, drift_module, diffusion_module, "diagonal", "ito"
        )
