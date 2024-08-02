import dataclasses
from typing import Sequence, Callable
import torch
import torch.nn as nn

type module_type = Callable[[], nn.Module]


@dataclasses.dataclass
class NetworkArchitecture:
    activation: module_type
    time_tail_sizes: Sequence[int]
    space_tail_sizes: Sequence[int]
    hidden_head_sizes: Sequence[int]


default_time_homogenous_architecture = NetworkArchitecture(
    activation=nn.Softplus,
    time_tail_sizes=(),
    space_tail_sizes=(),
    hidden_head_sizes=(64, 64)
)

default_time_heterogenous_architecture = NetworkArchitecture(
    activation=nn.Softplus,
    time_tail_sizes=(64,),
    space_tail_sizes=(64,),
    hidden_head_sizes=(64,)
)


class SeqSoftplusNet(nn.Sequential):
    """
    A certificate network
    """

    def __init__(self,
                 activation: module_type = nn.Softplus,
                 sizes: Sequence[int] = (64, 64)
                 ):
        layers = ((nn.LazyLinear(size), activation()) for size in sizes)
        flattened_layers = (spec for layer in layers for spec in layer)
        super().__init__(*flattened_layers)


class CertificateNet(nn.Module):
    def __init__(self,
                 time_homogenous: bool = True,
                 architecture: str | NetworkArchitecture = "default"
                 ):
        super().__init__()
        self.time_homogenous = time_homogenous
        if architecture == "default":
            if time_homogenous:
                architecture = default_time_homogenous_architecture
            else:
                architecture = default_time_heterogenous_architecture
        activation = architecture.activation
        self.time_tail = SeqSoftplusNet(
            activation,
            architecture.time_tail_sizes
        )
        self.space_tail = SeqSoftplusNet(
            activation,
            architecture.space_tail_sizes
        )
        head_sizes = architecture.hidden_head_sizes + (1,)
        self.head = SeqSoftplusNet(activation, head_sizes)

    def forward(self, t, x):
        if not self.time_homogenous:
            assert t.shape[1] == 1, "time must be n-by-1 tensor"
            output_t_tail = self.time_tail(t)
            assert x.shape[0] == t.shape[0], \
                "time and state tensors must have the same number of rows"
            output_space_tail = self.space_tail(x)

            output = torch.cat((output_t_tail, output_space_tail), 1)
            output = self.head(output)
            return output
        else:
            output_space_tail = self.space_tail(x)
            output = self.head(output_space_tail)
            return output
