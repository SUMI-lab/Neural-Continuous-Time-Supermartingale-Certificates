"""Provides a class for sampling Sobol sequences."""
from typing import Sequence
import numpy as np
import numpy.random as npr
from .sampler import Sampler


class RandomSampler(Sampler):
    """
    A class for sampling of random points in time [0, Inf) and
    space (l-dimensional boxes).
    """

    def __init__(self, low: Sequence[float], high: Sequence[float]):
        super().__init__(low, high)

    def sample_from_unit_box(self, n: int = 100) -> np.ndarray:
        return npr.uniform(
            low=self.low,
            high=self.high,
            size=(n, self.high.shape[0])
        )
