import torch
import torch.nn as nn
from controlled_sde import ControlledSDE


class CertificateModule(torch.nn.Sequential):
    """The RSA certificate"""

    def __init__(
        self,
        n_in: int = 2,
        n_out: int = 1,
        n_hidden: int = 32,
        device: torch.device | str = "cpu"
    ):
        super().__init__(
            nn.Linear(n_in, n_hidden, dtype=torch.float32, device=device),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, dtype=torch.float32, device=device),
            nn.Tanh(),
            nn.Linear(n_hidden, n_out, dtype=torch.float32, device=device),
            nn.Softplus()
        )

    def activation_derivative(self, x: torch.Tensor):
        """derivative of the activation function"""
        return 1.0 - torch.square(torch.tanh(x))

    def activation_second_derivative(self, x: torch.Tensor):
        """second derivative of the activation function"""
        return -2 * torch.tanh(x) * self.activation_derivative(x)

    # def activation_derivative_last(self, x: torch.Tensor):
    #     """derivative of the activation function"""
    #     return 1.0 - torch.square(torch.tanh(x))

    # def activation_second_derivative_last(self, x: torch.Tensor):
    #     """second derivative of the activation function"""
    #     return -2 * torch.tanh(x) * self.activation_derivative(x)

    def activation_derivative_last(self, x: torch.Tensor):
        """derivative of the activation function"""
        e = torch.exp(x)
        return e / (1.0 + e)

    def activation_second_derivative_last(self, x: torch.Tensor):
        """second derivative of the activation function"""
        e = torch.exp(x)
        return e / torch.square(1.0 + e)

    def forward(self, x: torch.Tensor):
        """forward call, must be nonnegative"""
        return super().forward(x)


class CertificateModuleWithDerivatives(torch.nn.Module):
    """The RSA wrapper that also computes the derivatives"""

    def __init__(self, module: CertificateModule):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor):
        """Forward call that additioanlly returns Jacobian and Hessian

        Args:
            x (torch.Tensor): input tensor

        Returns:
            a tuple of three tensors: (V(x), J_x V(x), H_x V(x)), that is
            the certificate value, its Jacobian, and its Hessian.
        """
        # Using Lemma 1 from
        # http://proceedings.mlr.press/v119/singla20a/singla20a.pdf.
        # The Jacobian is b3 defined as
        # b0 = w0
        # b1 = (w1 * sigma'(z0)) @ b0
        # b2 = (w2 * sigma'(z1)) @ b1
        # b3 = sigma'(z2) @ b2
        # For the hessian,
        # H = sum_{i=0}^{2} bi.T @ (f3i * sigma''(zi) * bi)
        # We need to compute f30, f31, f32
        # f10 = w1
        # f20 = (w2 * sigma'(z2)) @ f10 = (w2 * sigma'(z2)) @ w1
        # f21 = w2
        # f30 = sigma'(z2) @ f20
        # f31 = sigma'(z2) @ f21 = sigma'(z2) @ w2
        # f32 = 1
        # This can probably be automated in the future, but currently we got
        # an error when trying to implement it in a loop and passing it to
        # auto_LiRPA. This might even change before the final submission if
        # there is an auto_LiRPA update. For now do the calculations manually.
        # It is also easier to compare to the explicit formula.

        out = [x]  # find the layer outputs [a0=x, z1, a1, z2, a2, z3, a3]
        for layer in self.module:
            out.append(layer(out[-1]))

        z = [out[1], out[3]]  # outputs of linear layers, we need these

        # compute the activation derivatives
        dsigma = [self.module.activation_derivative(
            zi).unsqueeze(1) for zi in z]
        d2sigma = [self.module.activation_second_derivative(
            zi).unsqueeze(1) for zi in z]
        z.append(out[5])
        dsigma.append(
            self.module.activation_derivative_last(z[2]).unsqueeze(1)
        )
        d2sigma.append(
            self.module.activation_second_derivative_last(z[2]).unsqueeze(1)
        )
        # extract the weight matrices, add a dimension in front because of the
        # batch computation
        w = [self.module[i].weight.unsqueeze(0) for i in range(0, 5, 2)]

        # find Jacobians b[i] of z[i] w.r.t x
        b = [w[0]]
        for i in range(0, 2):
            b.append((w[i+1] * dsigma[i]) @ b[i])

        # final the final Jacobian
        jacobian = dsigma[2] @ b[2]

        # find Jacobians fij of z[i] with respect to a[j]
        f20 = (w[2] * dsigma[2]) @ w[1]
        f30 = dsigma[2] @ f20
        f31 = dsigma[2] @ w[2]

        # compute the Hessian, torch.permute(b[0], (0, 2, 1)) is transposing
        # dimensions 2 and 1 (dimension 0 is the batch indexing dimension)
        hessian = (f30 * d2sigma[0] * torch.permute(b[0], (0, 2, 1))) @ b[0] + \
            (f31 * d2sigma[1] * torch.permute(b[1], (0, 2, 1))) @ b[1] + \
            (d2sigma[2] * torch.permute(b[2], (0, 2, 1))) @ b[2]

        return out[-1], jacobian, hessian


class GeneratorModule(torch.nn.Module):
    """
    A module to compute the SDE's infinitesimal generator of a torch Module.
    """

    def __init__(self, certificate: CertificateModuleWithDerivatives, sde: ControlledSDE):
        super().__init__()
        self.policy = sde.policy
        self.drift = sde.drift
        self.diffusion = sde.diffusion
        self.certificate = certificate

    def forward(self, x: torch.Tensor):
        # Ideally, would like to call self.sde.generator(x, dv_dx, d2v_dx2)
        # but auto_LiRPA does not support this. Instead, extract from the SDE
        # the three networks (policy, drift, diffusion), and calculate the
        # generator here.
        u = self.policy(x)
        f = self.drift(x, u)
        g = self.diffusion(x, u)
        _, jacobian, hessian = self.certificate(x)
        dv_dx = jacobian.squeeze(1)
        d2v_dx2 = torch.cat(
            (hessian[:, 0, 0].unsqueeze(0), hessian[:, 1, 1].unsqueeze(0)),
            dim=0
        ).T  # extract the diagonal; tried torch.diagonal, LiRPA complains
        return (f * dv_dx + 0.5 * torch.square(g) * d2v_dx2).sum(dim=1)
