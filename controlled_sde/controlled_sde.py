"""Provides the base class for controlled SDEs."""

from abc import ABC, abstractmethod
import torch
import torch.autograd as ag
import torchsde
import torchsde.types
from .type_hints import tensor, tensors, vector, tensor_function


class ControlledSDE(ABC):
    """
    The base class for controlled SDEs.
    """

    def __init__(self, policy: torch.nn.Module,
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
        return self.policy(x)

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

    def generator(self,
                  f: torch.nn.Module,
                  time_homogenous: bool = True
                  ) -> tensor_function:
        """Infinitesimal generator of the SDE's Feller-Dynkin process.

        Args:
            t (Sequence[float] | torch.Tensor): times
            x (torch.Tensor): states

        Returns:
            torch.Tensor: the value of the generator at the point
        """
        # the neural certificate has one input, these helper functions
        # concatenate t and x if they are separate in the time-heterogenous case
        if time_homogenous:
            def f_single_arg(x):
                return self.f(None, x)

            def g_single_arg(x):
                return self.g(None, x)
        else:
            def f_single_arg(x):
                return self.f(x[:, 0], x[:, 1:])

            def g_single_arg(x):
                return self.g(x[:, 0], x[:, 1:])

        # the generator function to return
        def gen(x: tensor) -> tensor:

            if torch.numel(x) == 0:
                return x

            with torch.no_grad():
                f_value = f_single_arg(x)
                g_value = g_single_arg(x)

            # The alternative way to find gradient/hessian is:
            # jacobian = torch.vmap(torch.func.jacfwd(f))
            # hessian = torch.vmap(ag.functional.hessian(f))
            # hessian_diag = torch.diagonal(
            #     hessian(x).squeeze(),
            #     dim1=-2,
            #     dim2=-1
            # )
            # nabla = jacobian(x)
            # for me, vjp works faster than the other method.
            _, vjpfunc = torch.func.vjp(f, x)
            vjps = vjpfunc(torch.ones((x.shape[0], 1), device=x.device))
            nabla = vjps[0]
            _, vjpfunc2 = torch.func.vjp(
                lambda x: vjpfunc(torch.ones((x.shape[0], 1), device=x.device))[0], x)
            vjps2 = vjpfunc2(torch.ones((x.shape[0], 1), device=x.device))
            hessian_diag = vjps2[0]
            g_value = (f_value * nabla).sum(dim=1) + 0.5 * \
                (torch.square(g_value) * hessian_diag).sum(dim=1)
            return g_value
        return gen

    @ torch.no_grad()
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

    @ abstractmethod
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
