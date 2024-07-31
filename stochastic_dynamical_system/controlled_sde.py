from abc import ABC, abstractmethod
import torch
import torchsde
import torchsde.types
from .type_hints import tensor, tensors, vector, policy_function


class ControlledSDE(ABC):
    def __init__(self, policy: policy_function, method: str,
                 noise_type: str, sde_type: str = "ito"):
        super(ControlledSDE, self).__init__()
        self.policy = policy
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.method = method

    @abstractmethod
    def drift(self, t: vector, x: tensor, u: tensor) -> tensor:
        pass

    @abstractmethod
    def diffusion(self, t: vector, x: tensor, u: tensor) -> tensor:
        pass

    def _get_u(self, t: vector, x: tensor):
        return torch.unsqueeze(self.policy(t, x), 1)

    def f(self, t: vector, x: tensor) -> tensor:
        u = self._get_u(t, x)
        return torch.cat(self.drift(t, x, u), dim=1)

    def g(self, t: vector, x: tensor) -> tensor:
        u = self._get_u(t, x)
        return torch.cat(self.diffusion(t, x, u), dim=1)

    @torch.no_grad()
    def sample(self, x0: tensor, ts: vector, u: tensor) -> tensor | tensors:
        return torchsde.sdeint(self, x0, ts, method=self.method)
