"""Provides the base class for controlled SDEs."""

from abc import ABC, abstractmethod
import torch
import torchsde
import torchsde.types
from .type_hints import tensor, tensors, vector, tensor_function


class ControlledSDE(ABC):
    """
    The base class for controlled SDEs.
    """

    def __init__(self, policy: tensor_function,
                 noise_type: str, sde_type: str = "ito"):
        super().__init__()
        self.policy = policy
        self.noise_type = noise_type
        self.sde_type = sde_type

    @abstractmethod
    def drift(self, t: vector, x: tensor, u: tensor) -> tensor:
        """
        For a process`d X_t(t, X_t, u) = f(t, X_t, u) dt + g(t, X_t, u) dW_t`
        the drift function `f` defines the deterministic part of the dynamics.

        Args:
            t (Sequence[float] | torch.Tensor): times
            x (torch.Tensor): states
            u (torch.Tensor): controls

        Returns:
            torch.Tensor: the values of the controlled drift `f(t, X_t, u)`
        """

    @abstractmethod
    def diffusion(self, t: vector, x: tensor, u: tensor) -> tensor:
        """
        For a process`d X_t(t, X_t, u) = f(t, X_t, u) dt + g(t, X_t, u) dW_t`
        the diffusion function `g` defines the stochastic part of the dynamics.

        Args:
            t (Sequence[float] | torch.Tensor): times
            x (torch.Tensor): states
            u (torch.Tensor): controls

        Returns:
            torch.Tensor: the values of the controlled diffusion `g(t, X_t, u)`
        """

    def _get_u(self, t: vector, x: tensor):
        return self.policy(t, x)

    def f(self, t: vector, x: tensor) -> tensor:
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
        u = self._get_u(t, x)
        return self.drift(t, x, u)

    def g(self, t: vector, x: tensor) -> tensor:
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
        u = self._get_u(t, x)
        return self.diffusion(t, x, u)

    @torch.no_grad()
    def sample(self, x0: tensor, ts: vector, method: str = "euler",
               dt: str | float = "auto", **kwargs) -> tensor | tensors:
        """
        For each value in `x0`, simulates a sample path issuing from that point.
        Values at times `ts` are returned, with the values at the first time
        equal to `x0`.

        Args:
            x0 (torch.Tensor): starting states of sample paths
            ts (Sequence[float] | torch.Tensor): times in non-decreasing order
            method (str, optional): Numerical solution method. See [torchsde
                documentation](https://github.com/google-research/torchsde/blob/master/DOCUMENTATION.md)
                for details. Also supports "analytical" if an analytical
                solution is provided. Defaults to "euler".
            dt (str | float, optional): time step for numerical solution. Either
                a number, or "auto" to try to infer automatically. Defaults to
                "auto".

        Returns:
            torch.Tensor | Sequence[torch.Tensor]: sample paths of the processes
                issuing at starting points `x0` at times `ts`.
        """  # pylint: disable=line-too-long
        if method == "analytical":
            return self.analytical_sample(x0, ts, **kwargs)
        if dt == "auto":
            dt = torch.max(ts).item() / 1e3
        return torchsde.sdeint(self, x0, ts, method=method, dt=dt)

    @abstractmethod
    def analytical_sample(self, x0: tensor, ts: vector, **kwargs):
        """
        For each value in `x0`, simulates a sample path issuing from that point
        using the analytical solution to the SDE. Values at times `ts` are
        returned, with the values at the first time equal to `x0`.

        Args:
            x0 (torch.Tensor): starting states of sample paths
            ts (Sequence[float] | torch.Tensor): times in non-decreasing order

        Returns:
            torch.Tensor | Sequence[torch.Tensor]: sample paths of the processes
                issuing at starting points `x0` at times `ts`.
        """

    def close(self):
        """Ensure that the SDE object is released properly."""
