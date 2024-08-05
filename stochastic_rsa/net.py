from typing import Sequence, Callable
import torch
import torch.nn as nn


type module_type = Callable[[], nn.Module]


class CustomFinalLayer(nn.Module):
    def forward(self, x):
        return torch.tanh(x) + 1.0


class CertificateNet(nn.Sequential):
    """
    A certificate network
    """

    def __init__(self,
                 activation: module_type = nn.Tanh,
                 sizes: Sequence[int] = (64, 64),
                 nonnegative_activation: module_type = CustomFinalLayer,
                 device: torch.device | str = "cpu"
                 ):
        layers = ((nn.LazyLinear(size, dtype=torch.float32, device=device), activation())
                  for size in sizes)
        flattened_layers = [spec for layer in layers for spec in layer]
        flattened_layers.append(nn.LazyLinear(
            1, dtype=torch.float32, device=device))
        flattened_layers.append(nonnegative_activation())
        super().__init__(*flattened_layers)

    def forward(self, x):
        if torch.numel(x) == 0:
            return x
        return super().forward(x)
