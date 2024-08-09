import torch
import auto_LiRPA
from auto_LiRPA.bound_general import BoundSqr


class TanhWithDerivatives(torch.nn.Linear):
    def forward(self, u: torch.Tensor, b: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor):
        z = super().forward(u)
        a = torch.tanh(z)
        weight = self.weight.T
        dsig = (1.0 - torch.square(a))
        d2sig = -2 * a * dsig

        # print(z.shape, f.shape, dsig.shape, d2sig.shape, weight.shape)
        multiplier = dsig.unsqueeze(1) * weight

        b_next = b @ multiplier
        f1_next = f1 @ multiplier
        f2_next = f2 @ multiplier

        return a, d2sig, b_next, f1_next, f2_next


class Policy(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
        )


class Drift(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = 9.81 / 0.5
        self.a2 = 6.0 / (0.15 * 0.5 * 0.5)
        self.a3 = 0.1 / (0.15 * 0.5 * 0.5)

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        phi, theta = torch.split(x, split_size_or_sections=(1, 1), dim=1)

        f_phi = self.a1 * torch.sin(theta) + self.a2 * u - self.a3 * phi
        f_theta = phi
        return torch.cat([f_phi, f_theta], dim=1)


class Diffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = 1.0

    def forward(self, x: torch.Tensor, _u: torch.Tensor):
        g_phi = torch.full((x.shape[0], 1), self.sigma, device=x.device)
        return torch.cat([g_phi, torch.zeros_like(g_phi)], dim=1)


class Certificate(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            TanhWithDerivatives(2, 32),
            TanhWithDerivatives(32, 32),
            TanhWithDerivatives(32, 1),
        )

    def forward(self, x: torch.Tensor, dx_dx: torch.Tensor,
                dummy_f1: torch.Tensor,
                dummy_f2: torch.Tensor
                ):
        a0, b0, f01, f00 = x, dx_dx, dummy_f1, dummy_f2
        a1, d2sig1, b1, f11, f10 = self[0](a0, b0, f01, f00)
        a2, d2sig2, b2, f21, f20 = self[1](a1, b1, f11, f10)
        a3, d2sig3, b3, f31, f30 = self[2](a2, b2, f21, f20)
        jacobian = b3.squeeze(-1)
        hessian = b1 @ (b1.transpose(1, 2) * (d2sig1.unsqueeze(-1) * f31))
        hessian += b2 @ (b2.transpose(1, 2) * (d2sig2.unsqueeze(-1) * f30))
        hessian += b3 @ (b3.transpose(1, 2) * d2sig3.unsqueeze(-1))
        second_derivative = torch.cat(
            (hessian[:, 0, 0].unsqueeze(0), hessian[:, 1, 1].unsqueeze(0)), dim=0).T
        return a3 + 1.0, jacobian, second_derivative


class GeneratorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = Policy()
        self.drift = Drift()
        self.diffusion = Diffusion()
        self.certificate = Certificate()

    def forward(self, x: torch.Tensor, dx_dx, f1, f2):
        u = self.policy(x)
        f = self.drift(x, u)
        g = self.diffusion(x, u)
        v, dv_dx, d2v_dx2 = self.certificate(x, dx_dx, f1, f2)
        # = self.certificate(dv_dx, )
        # print(f.shape, g.shape, v.shape, dv_dx.shape)
        return (f * dv_dx + 0.5 * torch.square(g) * d2v_dx2).sum(dim=1)


x = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.float32)
dx_dx = torch.tile(torch.eye(2).unsqueeze(0), (3, 1, 1))
# d2x_dx2 = torch.zeros((3, 1, 2, 2))
dummy_x = torch.empty_like(x, dtype=torch.float32)
perturbation = auto_LiRPA.PerturbationLpNorm(eps=0.5)
zero_perturbation = auto_LiRPA.PerturbationLpNorm(eps=0.0)
x_bounded = auto_LiRPA.BoundedTensor(x, perturbation)
dx_dx_bounded = auto_LiRPA.BoundedTensor(dx_dx, zero_perturbation)
# d2x_dx2_bounded = auto_LiRPA.BoundedTensor(d2x_dx2, zero_perturbation)
f1 = torch.zeros((3, 32, 2))
f2 = torch.zeros((3, 32, 2))
f1_bounded = auto_LiRPA.BoundedTensor(f1, zero_perturbation)
f2_bounded = auto_LiRPA.BoundedTensor(f2, zero_perturbation)
gen = GeneratorModule()
m_bounded = auto_LiRPA.BoundedModule(
    gen, (dummy_x, torch.empty_like(dx_dx), f1, f2))

print(m_bounded.compute_bounds(
    (x_bounded, dx_dx_bounded, f1_bounded, f2_bounded), method="IBP"))
