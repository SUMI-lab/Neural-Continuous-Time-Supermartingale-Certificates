import torch
from .controlled_sde import ControlledSDE
from .type_hints import policy_function


class DriftingBarrier(ControlledSDE):
    """
    Stochastic drifting barrier.
    """

    def __init__(self, policy: policy_function, drift: float = 0.4):
        super(DriftingBarrier, self).__init__(
            policy, "srk", "diagonal", "ito")
        self.a = drift

    def drift(self, t, x, u):
        return self.a * x

    def diffusion(self, t, x, u):
        return u
