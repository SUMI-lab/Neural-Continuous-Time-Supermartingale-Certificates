from typing import Sequence, Callable
import torch
import torch.nn as nn


type module_type = Callable[[], nn.Module]


class TanhLayerWithDerivatives(nn.Linear):
    def forward(self, u, dudx=None, d2u_dx2=None, return_derivatives=False):
        z = super().forward(u)
        f = torch.tanh(z)
        if not return_derivatives:
            return f, None, None
        weight = self.weight.T
        dsig = 1.0 - torch.square(f)
        d2sig = -2.0 * dsig * f

        # print(z.shape, f.shape, dsig.shape, d2sig.shape, weight.shape)
        dfdx = dsig.unsqueeze(1) * weight

        d2fd2x = (d2sig.unsqueeze(1) * weight).unsqueeze(2) * \
            weight.unsqueeze(0).unsqueeze(0)
        if d2u_dx2 is not None:
            d2fd2x = dudx.unsqueeze(
                1) @ (d2fd2x.permute(0, 3, 1, 2) @ dudx.permute(0, 2, 1).unsqueeze(1))
            d2fd2x = d2fd2x.permute(0, 2, 3, 1)

            # extra term to compute full Hessian
            d2fd2x += d2u_dx2 @ dfdx.unsqueeze(1)

        if dudx is not None:
            dfdx = dudx @ dfdx
        # print(f"shape is: {dfdx.shape}")
        return f, dfdx, d2fd2x


class TanhLayerPlusOneWithDerivatives(TanhLayerWithDerivatives):
    def forward(self, u, dudx=None, d2f_dx2=None, return_derivatives=False):
        f, dfdx, d2f_dx2 = super().forward(u, dudx, d2f_dx2, return_derivatives)
        return f + 1.0, dfdx, d2f_dx2


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

    def forward(self, u, dudx=None, d2u_dx2=None, return_derivatives=False):
        if torch.numel(u) == 0:
            return u
        f = u
        dfdx = dudx if return_derivatives else None
        d2f_dx2 = d2u_dx2 if return_derivatives else None
        for layer in self.layers:
            f, dfdx, d2f_dx2 = layer.forward(
                f, dfdx, d2f_dx2, return_derivatives)
        f, dfdx, d2f_dx2 = self.final_layer(
            f, dfdx, d2f_dx2, return_derivatives)
        if return_derivatives:
            dfdx = dfdx.squeeze(-1)
            d2f_dx2 = torch.diagonal(d2f_dx2.squeeze(-1), dim1=-2, dim2=-1)
            return f, dfdx.squeeze(-1), d2f_dx2
        else:
            return f
