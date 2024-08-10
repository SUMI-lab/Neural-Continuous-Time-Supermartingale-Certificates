"""Provides the base class for controlled SDEs."""
from abc import ABC
import torch
import torchsde
import torchsde.types


class ControlledSDE(ABC):
    """
    The base class for controlled SDEs.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        drift: torch.nn.Module,
        diffusion: torch.nn.Module,
        noise_type: str,
        sde_type: str = "ito"
    ):
        super().__init__()
        self.policy = policy
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.drift = drift
        self.diffusion = diffusion

    def _get_u(self, x: torch.Tensor):
        return self.policy(x)

    def f(self, _t, x: torch.Tensor) -> torch.Tensor:
        """
        For a process`d X_t(t, X_t, u) = f(t, X_t, u) dt + g(t, X_t, u) dW_t`
        returns the drift `f_pi(t, X_t) = f(t, X_t, pi(t, X_t))` under the
        policy. The name `f` is needed for `torchsde` to identify it.

        Args:
            t (Sequence[float] | torch.Tensor): times
            x (torch.Tensor): states

        Returns:
            torch.Tensor: drift `f_pi(t, X_t)` under the policy `pi`
        """
        u = self._get_u(x)
        return self.drift(x, u)

    def g(self, _t, x: torch.Tensor) -> torch.Tensor:
        """
        For a process`d X_t(t, X_t, u) = f(t, X_t, u) dt + g(t, X_t, u) dW_t`
        returns the diffusion `g_pi(t, X_t) = g(t, X_t, pi(t, X_t))` under the
        policy. The name `g` is needed for `torchsde` to identify it.

        Args:
            t (Sequence[float] | torch.Tensor): times
            x (torch.Tensor): states

        Returns:
            torch.Tensor: diffusion `g_pi(t, X_t)` under the policy `pi`
        """
        u = self._get_u(x)
        return self.diffusion(x, u)

    def generator(self,
                  x: torch.Tensor,
                  gradient: torch.Tensor,
                  hessian_diagonal: torch.Tensor
                  ) -> torch.Tensor:
        """Infinitesimal generator of the SDE's Feller-Dynkin process.

        Args:
            x (torch.Tensor): states
            gradient (torch.Tensor): vector of first derivative values
            hessian_diagonal (torch.Tensor): vector of second derivatives

        Returns:
            torch.Tensor: the value of the generator at the point
        """
        f = self.f(None, x)
        g = self.g(None, x)
        generator_value = (
            f * gradient + 0.5 * torch.square(g) * hessian_diagonal
        ).sum(dim=-1)
        return generator_value

    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor,
        times: torch.Tensor,
        method: str = "euler",
        dt: str | float = "auto",
        **kwargs
    ):
        """
        For each value in `x0`, simulates a sample path issuing from that point.
        Values at times `ts` are returned, with the values at the first time
        equal to `x0`.

        Args:
            x0 (torch.Tensor): starting states of sample paths
            ts (Sequence[float] | torch.Tensor): times in non-decreasing order
            method (str, optional): Numerical solution method. See [torchsde
                documentation](https://github.com/google-research/torchsde/blob/master/DOCUMENTATION.md)
                for details. Defaults to "euler".
            dt (str | float, optional): time step for numerical solution. Either
                a number, or "auto" to try to infer automatically. Defaults to
                "auto".

        Returns:
            torch.Tensor | Sequence[torch.Tensor]: sample paths of the processes
                issuing at starting points `x0` at times `ts`.
        """  # pylint: disable=line-too-long
        if dt == "auto":
            dt = torch.max(times).item() / 1e3
        return torchsde.sdeint(self, x0, times, method=method, dt=dt)

    def close(self):
        """Ensure that the SDE object is released properly."""
