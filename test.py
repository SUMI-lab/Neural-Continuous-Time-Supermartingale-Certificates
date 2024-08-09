import torch
import auto_LiRPA


class TanhWithDerivatives(torch.nn.Linear):
    def forward(self, u: torch.Tensor, dudx: torch.Tensor):
        f = torch.tanh(super().forward(u))
        weight = self.weight.T
        dsig = 1.0 - torch.square(f)
        # d2sig = -2.0 * dsig * f

        # print(z.shape, f.shape, dsig.shape, d2sig.shape, weight.shape)
        dfdx = dsig.unsqueeze(1) * weight
        dfdx = dudx @ dfdx

        # d2fd2x = (d2sig.unsqueeze(1) * weight).unsqueeze(2) * \
        #     weight.unsqueeze(0).unsqueeze(0)

        # if dudx is not None:
        #     d2fd2x = dudx.unsqueeze(1) @ \
        #         (d2fd2x.permute(0, 3, 1, 2) @
        #          dudx.permute(0, 2, 1).unsqueeze(1))
        #     d2fd2x = d2fd2x.permute(0, 2, 3, 1)
        #     # print(dfdx.unsqueeze(1).shape)
        #     if d2u_dx2 is not None:
        #         d2fd2x += d2u_dx2 @ dfdx.unsqueeze(1)

        # print(f"shape is: {dfdx.shape}")
        return f, dfdx


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

    def forward(self, x: torch.Tensor):
        v = x
        n_samples = x.shape[0]
        dv_dx = torch.tile(torch.eye(2).unsqueeze(0), (n_samples, 1, 1))
        for layer in self.children():
            v, dv_dx = layer(v, dv_dx)
        return v, dv_dx


class GeneratorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = Policy()
        self.drift = Drift()
        self.diffusion = Diffusion()
        self.certificate = Certificate()

    def forward(self, x: torch.Tensor):
        u = self.policy(x)
        f = self.drift(x, u)
        g = self.diffusion(x, u)
        v, dv_dx = self.certificate(x)
        print(f.shape, g.shape, v.shape, dv_dx.shape)
        return (f + g).sum(dim=1) + v.sum(dim=1)


x = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.float32)
dummy_x = torch.empty_like(x, dtype=torch.float32)
perturbation = auto_LiRPA.PerturbationLpNorm(eps=0.5)
x_bounded = auto_LiRPA.BoundedTensor(x, perturbation)
gen = GeneratorModule()
m_bounded = auto_LiRPA.BoundedModule(gen, (dummy_x,))

print(m_bounded.compute_bounds(x_bounded))
