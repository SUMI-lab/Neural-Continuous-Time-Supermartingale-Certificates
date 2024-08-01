"""Provides the drifting barrier SDE."""

from .controlled_sde import ControlledSDE
from .type_hints import policy_function


class DriftingBarrier(ControlledSDE):
    """
    Stochastic drifting barrier SDE from the paper.
    """

    def __init__(self, policy: policy_function, drift: float = 0.4):
        super().__init__(policy, "diagonal", "ito")
        self.a = drift

    def drift(self, _t, x, _u):
        return self.a * x

    def diffusion(self, _t, _x, u):
        return u

    def analytical_sample(self, _x0, _ts, **kwargs):
        raise NotImplementedError
