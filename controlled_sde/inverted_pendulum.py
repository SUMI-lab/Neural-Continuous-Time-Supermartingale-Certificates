import torch
from .controlled_sde import ControlledSDE
from .type_hints import policy_function, tensor
import numpy as np


class InvertedPendulum(ControlledSDE):
    """
    Stochastic inverted pendulum.

    The control signal is normalized (i.e. in [-1,1]) and multiplied by
    the maximum torque afterwards.
    """

    def __init__(self, policy: policy_function, maximum_torque: float = 6.0,
                 pendulum_length: float = 0.5, ball_mass: float = 0.15,
                 friction: float = 0.1, gravity: float = 9.81,
                 volatility_scale: float = 1.0):
        super(InvertedPendulum, self).__init__(
            policy, "diagonal", "ito")
        self.a1 = gravity / pendulum_length
        self.a2 = maximum_torque / ball_mass / (pendulum_length ** 2)
        self.a3 = friction / ball_mass / (pendulum_length ** 2)
        self.sigma = volatility_scale

        self.screen_dim = 500
        self.screen = None
        self.clock = None

    def drift(self, t, x, u):
        phi, theta = torch.split(x, split_size_or_sections=(1, 1), dim=1)

        f_phi = self.a1 * torch.sin(theta) + self.a2 * u - self.a3
        f_theta = phi
        return torch.cat([f_phi, f_theta], dim=1)

    def diffusion(self, t, x, u):
        phi, _ = torch.split(x, split_size_or_sections=(1, 1), dim=1)
        g_phi = self.sigma * phi
        return torch.cat([g_phi, torch.zeros_like(g_phi)], dim=1)

    def analytical_sample(self, x0, ts, int_f, int_g):
        raise NotImplementedError(
            ("No analytical solution exists for stochastic inverted pendulum; "
             "please, use another method instead, for example, 'euler'.")
        )

    def render(self, sample_paths: tensor, ts: tensor):
        # Based on Gymnasium's implementation for PendulumEnv
        try:
            import pygame
            from pygame import gfxdraw
        except:
            raise ImportError(
                "pygame is not installed, run `pip install pygame`")

        batch_size = sample_paths.shape[1]
        n_per_axis = int(np.ceil(np.sqrt(batch_size)))
        bound = 1.5
        scale = self.screen_dim / (bound * 2) / n_per_axis

        for time, time_next, angles in zip(ts, torch.roll(ts, -1), sample_paths[:, :, 1].squeeze()):
            if self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
            self.surf.fill((255, 255, 255))

            for i in range(batch_size):
                state = angles[i]

                x, y = divmod(i, n_per_axis)
                offset_x = self.screen_dim * (2 * x + 1) // (2 * n_per_axis)
                offset_y = self.screen_dim * (2 * y + 1) // (2 * n_per_axis)

                rod_length = 1 * scale
                rod_width = 0.2 * scale
                l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2

                coords = [(l, b), (l, t), (r, t), (r, b)]
                transformed_coords = []
                for c in coords:
                    c = pygame.math.Vector2(c).rotate_rad(
                        state.item() + np.pi / 2)
                    c = (c[0] + offset_x, c[1] + offset_y)
                    transformed_coords.append(c)
                gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
                gfxdraw.filled_polygon(
                    self.surf, transformed_coords, (204, 77, 77))

                gfxdraw.aacircle(self.surf, offset_x, offset_y,
                                 int(rod_width / 2), (204, 77, 77))
                gfxdraw.filled_circle(
                    self.surf, offset_x, offset_y, int(
                        rod_width / 2), (204, 77, 77)
                )

                rod_end = (rod_length, 0)
                rod_end = pygame.math.Vector2(
                    rod_end).rotate_rad(state.item() + np.pi / 2)
                rod_end = (int(rod_end[0] + offset_x),
                           int(rod_end[1] + offset_y))
                gfxdraw.aacircle(
                    self.surf, rod_end[0], rod_end[1], int(
                        rod_width / 2), (204, 77, 77)
                )
                gfxdraw.filled_circle(
                    self.surf, rod_end[0], rod_end[1], int(
                        rod_width / 2), (204, 77, 77)
                )

                # drawing axle
                gfxdraw.aacircle(self.surf, offset_x, offset_y,
                                 int(0.05 * scale), (0, 0, 0))
                gfxdraw.filled_circle(self.surf, offset_x, offset_y,
                                      int(0.05 * scale), (0, 0, 0))

            self.surf = pygame.transform.flip(self.surf, False, True)
            self.screen.blit(self.surf, (0, 0))

            pygame.event.pump()

            if time_next > time:
                self.clock.tick(1/(time_next - time))
            pygame.display.flip()

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
