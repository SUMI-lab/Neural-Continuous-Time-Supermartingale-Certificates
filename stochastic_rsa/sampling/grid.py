"""Provides a class for sampling points on multidimensional grids."""
import numpy as np
from .sampler import Sampler


class GridSampler(Sampler):
    """
    A class for sampling of points in time [0, Inf) and
    space (l-dimensional boxes) using grids for space.
    """

    def sample_from_unit_box(self, n: int = 100) -> np.ndarray:
        n_dim = self.n_dim
        n_points_per_dim = int(np.ceil(np.power(n, 1.0 / n_dim)))
        points_along_one_dimension = np.linspace(0, 1, n_points_per_dim)
        grid = np.meshgrid(*((points_along_one_dimension,) * n_dim))
        grid = np.concatenate(grid).reshape(
            (n_dim, n_points_per_dim ** n_dim)
        ).transpose()
        return grid[:n, :]
