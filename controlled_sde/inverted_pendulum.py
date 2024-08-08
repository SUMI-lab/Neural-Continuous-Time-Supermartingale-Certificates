"""Provides the inverted pendulum SDE."""

import dataclasses
import torch
import numpy as np
from .controlled_sde import ControlledSDE
from .type_hints import tensor_function, tensor


@dataclasses.dataclass
class RenderingData:
    "Rendering data for pygame implementation of the animation."
    screen_dim: int
    screen: object
    clock: object
    surf: object


@dataclasses.dataclass
class PendulumData:
    "Parameters of the pendulum."
    maximum_torque: float
    pendulum_length: float
    ball_mass: float
    friction: float


default_pendulum_data = PendulumData(
    maximum_torque=6.0,
    pendulum_length=0.5,
    ball_mass=0.15,
    friction=0.1
)


class InvertedPendulum(ControlledSDE):
    """
    Stochastic inverted pendulum.

    The control signal is normalized (i.e. in [-1,1]) and multiplied by
    the maximum torque afterwards.
    """

    def __init__(self, policy: torch.nn.Module,
                 pendulum_data: PendulumData = default_pendulum_data,
                 gravity: float = 9.81,
                 volatility_scale: float = 2.0):
        super().__init__(policy, "diagonal", "ito")

        # Precompute auxiliary constants
        self.a1 = gravity / pendulum_data.pendulum_length
        denom = pendulum_data.ball_mass * (pendulum_data.pendulum_length ** 2)
        self.a2 = pendulum_data.maximum_torque / denom
        self.a3 = pendulum_data.friction / denom
        self.sigma = volatility_scale

        # Initialize rendering variables
        self.rendering_data = RenderingData(
            screen_dim=500, screen=None, clock=None, surf=None)

    def drift(self, x, u):
        phi, theta = torch.split(x, split_size_or_sections=(1, 1), dim=1)

        f_phi = self.a1 * torch.sin(theta) + self.a2 * u - self.a3 * phi
        f_theta = phi
        return torch.cat([f_phi, f_theta], dim=1)

    def diffusion(self, x, _u):
        # phi, _ = torch.split(x, split_size_or_sections=(1, 1), dim=1)
        # g_phi = self.sigma * phi
        g_phi = torch.full((x.shape[0], 1), self.sigma, device=x.device)
        return torch.cat([g_phi, torch.zeros_like(g_phi)], dim=1)

    def analytical_sample(self, _x0, _ts, **kwargs):
        raise NotImplementedError(
            ("No analytical solution exists for stochastic inverted pendulum; "
             "please, use another method instead, for example, 'euler'.")
        )

    def render(self, sample_paths: tensor, ts: tensor):
        # Based on Gymnasium's implementation for PendulumEnv
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise ImportError(
                "pygame is not installed, run `pip install pygame`"
            ) from e

        sample_paths = sample_paths.cpu()
        ts = ts.cpu()

        screen_dim = self.rendering_data.screen_dim
        screen = self.rendering_data.screen
        clock = self.rendering_data.clock
        surf = self.rendering_data.surf

        batch_size = sample_paths.shape[1]
        n_per_axis = int(np.ceil(np.sqrt(batch_size)))
        bound = 1.5
        scale = screen_dim / (bound * 2) / n_per_axis

        for time, time_next, angles in zip(ts, torch.roll(ts, -1), sample_paths[:, :, 1].squeeze()):

            if screen is None:
                pygame.init()
                pygame.display.init()
                screen = pygame.display.set_mode((screen_dim, screen_dim))
            if clock is None:
                clock = pygame.time.Clock()

            surf = pygame.Surface((screen_dim, screen_dim))
            surf.fill((255, 255, 255))

            for i in range(batch_size):
                state = angles[i]

                y, x = divmod(i, n_per_axis)
                y = n_per_axis - y - 1
                offset_x = screen_dim * (2 * x + 1) // (2 * n_per_axis)
                offset_y = screen_dim * (2 * y + 1) // (2 * n_per_axis)

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
                gfxdraw.aapolygon(surf, transformed_coords, (204, 77, 77))
                gfxdraw.filled_polygon(surf, transformed_coords, (204, 77, 77))

                gfxdraw.aacircle(surf, offset_x, offset_y,
                                 int(rod_width / 2), (204, 77, 77)
                                 )
                gfxdraw.filled_circle(surf, offset_x, offset_y,
                                      int(rod_width / 2), (204, 77, 77)
                                      )

                rod_end = (rod_length, 0)
                rod_end = pygame.math.Vector2(rod_end).rotate_rad(
                    state.item() + np.pi / 2)
                rod_end = (int(rod_end[0] + offset_x),
                           int(rod_end[1] + offset_y)
                           )
                gfxdraw.aacircle(
                    surf, rod_end[0], rod_end[1],
                    int(rod_width / 2), (204, 77, 77)
                )
                gfxdraw.filled_circle(
                    surf, rod_end[0], rod_end[1],
                    int(rod_width / 2), (204, 77, 77)
                )

                # drawing axle
                gfxdraw.aacircle(surf, offset_x, offset_y,
                                 int(0.05 * scale), (0, 0, 0)
                                 )
                gfxdraw.filled_circle(surf, offset_x, offset_y,
                                      int(0.05 * scale), (0, 0, 0)
                                      )

            surf = pygame.transform.flip(surf, False, True)
            screen.blit(surf, (0, 0))

            pygame.event.pump()

            if time_next > time:
                clock.tick(1/(time_next - time))
            pygame.display.flip()

    def close(self):
        if self.rendering_data.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
