from typing import Sequence, Callable
import torch
import torch.nn as nn


type module_type = Callable[[], nn.Module]


class TanhLayerWithDerivatives(nn.Linear):
    def forward(self, x):
        return torch.tanh(super().forward(x))

    def forward_derivatives(self, u, dudx, d2u_dx2):
        f = self.forward(u)
        weight = self.weight.T
        dsig = 1.0 - torch.square(f)
        d2sig = -2.0 * dsig * f

        # print(z.shape, f.shape, dsig.shape, d2sig.shape, weight.shape)
        dfdx = dsig.unsqueeze(1) * weight

        d2fd2x = (d2sig.unsqueeze(1) * weight).unsqueeze(2) * \
            weight.unsqueeze(0).unsqueeze(0)

        if dudx is not None:
            d2fd2x = dudx.unsqueeze(1) @ \
                (d2fd2x.permute(0, 3, 1, 2) @
                 dudx.permute(0, 2, 1).unsqueeze(1))
            d2fd2x = d2fd2x.permute(0, 2, 3, 1)
            # print(dfdx.unsqueeze(1).shape)
            if d2u_dx2 is not None:
                d2fd2x += d2u_dx2 @ dfdx.unsqueeze(1)

            dfdx = dudx @ dfdx
        # print(f"shape is: {dfdx.shape}")
        return f, dfdx, d2fd2x


class TanhLayerPlusOneWithDerivatives(TanhLayerWithDerivatives):
    def forward(self, x):
        f = super().forward(x)
        return f + 1.0


class CertificateNet(nn.Module):
    """
    A certificate network
    """

    def __init__(self,
                 n_in: int = 2,
                 sizes: Sequence[int] = (64, 64),
                 device: torch.device | str = "cpu"
                 ):
        super().__init__()
        prev_sizes = (n_in, ) + sizes[:-1]
        self.layers = nn.ModuleList(
            TanhLayerWithDerivatives(in_features=prev_size, out_features=size)
            for (prev_size, size) in zip(prev_sizes, sizes)
        )
        self.final_layer = TanhLayerPlusOneWithDerivatives(
            in_features=sizes[-1], out_features=1
        )

    def forward(self, u):
        for layer in self.layers:
            u = layer(u)
        u = self.final_layer(u)
        return u

    def find_derivatives(self, u, dfdx, d2f_dx2):
        f = u
        n_dim = u.shape[1]
        for layer in self.layers:
            f, dfdx, d2f_dx2 = layer.forward_derivatives(f, dfdx, d2f_dx2)
        f, dfdx, d2f_dx2 = self.final_layer.forward_derivatives(
            f, dfdx, d2f_dx2
        )
        dfdx = dfdx.squeeze(-1)
        d2f_dx2 = torch.diagonal(d2f_dx2.squeeze(-1), dim1=-2, dim2=-1)
        return f, dfdx.squeeze(-1), d2f_dx2
