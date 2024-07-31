from abc import ABC, abstractmethod
import torch
import torchsde
import torchsde.types
from .type_hints import tensor, tensors, vector, policy_function, policy_integral


class ControlledSDE(ABC):
    def __init__(self, policy: policy_function,
                 noise_type: str, sde_type: str = "ito"):
        super(ControlledSDE, self).__init__()
        self.policy = policy
        self.noise_type = noise_type
        self.sde_type = sde_type

    @abstractmethod
    def drift(self, t: vector, x: tensor, u: tensor) -> tensor:
        pass

    @abstractmethod
    def diffusion(self, t: vector, x: tensor, u: tensor) -> tensor:
        pass

    def _get_u(self, t: vector, x: tensor):
        return self.policy(t, x)

    def f(self, t: vector, x: tensor) -> tensor:
        u = self._get_u(t, x)
        return self.drift(t, x, u)

    def g(self, t: vector, x: tensor) -> tensor:
        u = self._get_u(t, x)
        return self.diffusion(t, x, u)

    @torch.no_grad()
    def sample(self, x0: tensor, ts: vector,
               method: str = "euler", dt: str | float = "auto",
               int_f: policy_integral = None, int_g: policy_integral = None) -> tensor | tensors:
        if method == "analytical":
            return self.analytical_sample(x0, ts, int_f, int_g)
        if dt == "auto":
            dt = torch.max(ts).item() / 1e3
        return torchsde.sdeint(self, x0, ts, method=method, dt=dt)

    @abstractmethod
    def analytical_sample(self, x0: tensor, ts: vector, int_f: policy_integral, int_g: policy_integral):
        pass
