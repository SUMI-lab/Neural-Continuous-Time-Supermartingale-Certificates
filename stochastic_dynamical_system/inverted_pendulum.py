import torch
import torchsde
from .controlled_sde import ControlledSDE


class InvertedPendulum(ControlledSDE):
    """
    Stochastic inverted pendulum.

    The control signal is normalized (i.e. in [-1,1]) and multiplied by
    the maximum torque afterwards.
    """

    def __init__(self, pendulum_length: float = 0.5, ball_mass: float = 0.15,
                 friction: float = 0.1, maximum_torque: float = 6.0,
                 gravity: float = 9.81, volatility_scale: float = 0.1):
        super(InvertedPendulum, self).__init__("diagonal", "ito")
        self.a1 = gravity / pendulum_length
        self.a2 = maximum_torque / ball_mass / (pendulum_length ** 2)
        self.a3 = friction / ball_mass / (pendulum_length ** 2)
        self.sigma = volatility_scale

    def drift(self, t, x, u):
        phi, theta = torch.split(x, split_size_or_sections=(1, 1), dim=1)

        f_phi = self.a1 * torch.sin(theta) + self.a2 * u - self.a3
        f_theta = phi
        return f_phi, f_theta

    def diffusion(self, t, x, u):
        phi, _ = torch.split(x, split_size_or_sections=(1, 1), dim=1)
        g_phi = self.sigma * phi
        return g_phi, torch.zeros_like(g_phi)
