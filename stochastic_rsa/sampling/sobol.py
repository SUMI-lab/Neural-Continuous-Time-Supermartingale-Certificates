"""Provides a class for sampling Sobol sequences."""
from typing import Sequence
import numpy as np
import scipy.stats.qmc as qmc
from .sampler import Sampler


class SobolSampler(Sampler):
    """
    A class for sampling of points in time [0, Inf) and
    space (l-dimensional boxes) using Sobol sequences for space.
    """

    def __init__(self, low: Sequence[float], high: Sequence[float]):
        super().__init__(low, high)
        self.qmc_sampler = qmc.Sobol(self.n_dim, scramble=False)

    def sample_from_unit_box(self, n: int = 100) -> np.ndarray:
        # By default, Sobol sequence does not include the upper bound,
        # we can fix this by rescaling appropriately.
        pow2 = 2 ** np.ceil(np.log2(n))  # power of two greater or equal to n
        scaling_factor = pow2 / (pow2 - 1)

        return self.qmc_sampler.random(n) * scaling_factor
