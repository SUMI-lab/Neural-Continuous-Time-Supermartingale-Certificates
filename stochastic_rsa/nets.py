import torch
from controlled_sde import ControlledSDE


class TanhWithDerivatives(torch.nn.Linear):
    def forward(self, u: torch.Tensor, b: torch.Tensor, f: torch.Tensor):
        # print(f"in: {u.shape}")
        z = super().forward(u)
        a = torch.tanh(z)
        weight = self.weight.T.unsqueeze(0)
        dsig = (1.0 - torch.square(a))
        d2sig = -2 * a * dsig

        # print(z.shape, f.shape, dsig.shape, d2sig.shape, weight.shape)
        multiplier = dsig.unsqueeze(1) * weight
        # print(f"out: {z.shape}, {a.shape}, {dsig.shape}")
        # print(f"weights: {weight.shape}")
        if b is not None:
            # print(f"b: {b.shape}, {multiplier.shape}")
            b_next = b @ multiplier
        else:
            b_next = multiplier
        if f is not None:
            # print(f"f1: {f1.shape}, {multiplier.shape}")
            f_next = f @ multiplier
        else:
            f_next = multiplier

        return a, dsig.unsqueeze(1), d2sig.unsqueeze(2), b_next, f_next


class CertificateModule(torch.nn.Sequential):
    def __init__(self, device: torch.device):
        super().__init__(
            TanhWithDerivatives(2, 32, device=device),
            TanhWithDerivatives(32, 32, device=device),
            TanhWithDerivatives(32, 1, device=device),
        )

    def forward(self, x: torch.Tensor):
        for layer in self:
            x, _, _, _, _ = layer(x, None, None)
        return x + 1.0


class CertificateModuleWithDerivatives(torch.nn.Module):
    def __init__(self, module: CertificateModule):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor):
        # Using Lemma 1 from
        # http://proceedings.mlr.press/v119/singla20a/singla20a.pdf.
        # We need need f41, f42, f43
        # f41 = f31 @ sig3
        # f42 = f32 @ sig3
        # f43 = I
        # f31 = f21 @ (w3 * sig 2)
        # f32 = w3
        # f21 = w2

        a0 = x
        a1, _, sig1, _, _ = self.module[0](a0, None, None)
        b1 = self.module[0].weight.T.unsqueeze(0)
        a2, _, sig2, b2, _ = self.module[1](a1, b1, None)
        f21 = self.module[1].weight.T.unsqueeze(0)
        a3, sigma_final, sig3, b3, f31 = self.module[2](a2, b2, f21)
        b4 = b3 @ sigma_final
        f32 = self.module[2].weight.T.unsqueeze(0)
        f41 = f31 @ sigma_final
        f42 = f32 @ sigma_final

        jacobian = b4.squeeze(-1)
        hessian = b1 @ (b1.transpose(1, 2) * (sig1 * f41))
        hessian += b2 @ (b2.transpose(1, 2) * (sig2 * f42))
        hessian += b3 @ (b3.transpose(1, 2) * sig3)

        return a3 + 1.0, jacobian, hessian


class GeneratorModule(torch.nn.Module):
    def __init__(self, certificate: CertificateModuleWithDerivatives, sde: ControlledSDE):
        super().__init__()
        self.policy = sde.policy
        self.drift = sde.drift
        self.diffusion = sde.diffusion
        self.certificate = certificate

    def forward(self, x: torch.Tensor):
        u = self.policy(x)
        f = self.drift(x, u)
        g = self.diffusion(x, u)
        _, dv_dx, hessian = self.certificate(x)
        d2v_dx2 = torch.cat(
            (hessian[:, 0, 0].unsqueeze(0), hessian[:, 1, 1].unsqueeze(0)), dim=0).T
        return (f * dv_dx + 0.5 * torch.square(g) * d2v_dx2).sum(dim=1)
