from abc import ABC, abstractmethod
import torch
import torchsde
import torchsde.types
from .type_hints import tensor, tensors, vector


class ControlledSDE(ABC):
    def __init__(self, noise_type: str = "general", sde_type: str = "ito"):
        super(ControlledSDE, self).__init__()
        self.noise_type = noise_type
        self.sde_type = sde_type

    @abstractmethod
    def drift(self, t, x: tensor, u: tensor) -> tensor:
        pass

    @abstractmethod
    def diffusion(self, t, x: tensor, u: tensor) -> tensor:
        pass

    def f(self, t, x: tensor) -> tensor:
        u = 0
        return torch.cat(self.drift(t, x, u), dim=1)

    def g(self, t, x: tensor) -> tensor:
        u = 0
        return torch.cat(self.diffusion(t, x, u), dim=1)

    @torch.no_grad()
    def sample(self, x0: tensor, ts: vector, u: tensor) -> tensor | tensors:
        return torchsde.sdeint(self, x0, ts)
