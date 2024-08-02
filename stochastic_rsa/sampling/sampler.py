"""Provides a base class for point samplers."""
from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np


class Sampler(ABC):
    """
    A base class for samplers of points in time [0, Inf) and
    space (l-dimensional boxes).
    """

    def __init__(self, low: Sequence[float], high: Sequence[float]):
        len_low = len(low)
        len_high = len(high)
        assert len_low == len_high, (
            "lower and upper bounds must be the same length, ",
            f"got {len_low} and {len_high} instead."
        )
        self.n_dim = len_low
        self.low = np.array(low)
        self.high = np.array(high)
        self.magnitude = self.high-self.low
        super().__init__()

    def sample_time(self, n: int = 100, dt: float = 0.05) -> np.ndarray:
        """Samples `n` points starting from `0` with the distance between
        points starting at `dt` and growing exponentially.

        Args:
            n (int, optional): number of points. Defaults to 100.
            dt (float, optional): the initial distance. Defaults to 0.05.

        Returns:
            numpy.ndarray: the sampled points
        """
        ln_n = np.log(n)
        scale = dt / (ln_n - np.log(n-1))
        points = scale * (ln_n - np.log([n-i for i in range(n)]))
        return points

    def sample_space(self, n: int = 100) -> np.ndarray:
        """Samples `n` points in space given the space bounds.

        Args:
            n (int, optional): number of points. Defaults to 100.

        Returns:
            numpy.ndarray: `n` by `n_dim` array of sampled points
        """
        unscaled_points = self.sample_from_unit_box(n)
        return unscaled_points * self.magnitude + self.low

    @abstractmethod
    def sample_from_unit_box(self, n: int = 100) -> np.ndarray:
        """Samples `n` points in a unit box [0, 1]^`n_dim`.

        Args:
            n (int, optional): number of points. Defaults to 100.

        Returns:
            numpy.ndarray: `n` by `n_dim` array of sampled points
        """
