"""Provides a class for sampling Sobol sequences."""
from typing import Sequence
import numpy as np
import scipy.stats.qmc as qmc
from .sampler import Sampler


class SobolSampler(Sampler):
    """
    A base class for samplers of points in time [0, Inf) and
    space (l-dimensional boxes).
    """

    def __init__(self, low: Sequence[float], high: Sequence[float]):
        super().__init__(low, high)
        self.qmc_sampler = qmc.Sobol(self.n_dim, scramble=False)

    def sample_from_unit_box(self, n: int = 100) -> np.ndarray:
        return self.qmc_sampler.random(n)
