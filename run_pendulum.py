import torch
import torch.nn as nn
import numpy.random as npr
import matplotlib.pyplot as plt
import controlled_sde
from rl_agent import TanhPolicy


# Seed the random number generators
seeds = npr.randint(1, 1e5, size=(2,))
torch.manual_seed(seeds[0])
npr.seed(seeds[1])

DEVICE_STR = "cpu"  # Torch device
BATCH_SIZE = 16     # how many environments to run in parallel
DURATION = 60       # seconds
FPS = 20            # frames per second

STARTING_ANGLE = torch.pi    # starting angle for each sample path
STARTING_SPEED = 0.0         # starting angular velocity
T_SIZE = DURATION * FPS + 1  # number of time steps for each sample path


if DEVICE_STR == "auto":
    if torch.cuda.is_available() and torch.backends.cuda.is_built():
        DEVICE_STR = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE_STR = "mps"
    else:
        DEVICE_STR = "cpu"

device = torch.device(DEVICE_STR)


def policy_do_nothing(_t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A policy that performs zero-th action every time (does nothing).

    Args:
        _t (float | torch.Tensor): time
        x (torch.Tensor): state

    Returns:
        torch.Tensor: control (a vector of zeros)
    """
    return torch.zeros((x.size(0),), device=device).unsqueeze(1)


rl_policy_net = TanhPolicy(1, 64)
rl_policy_net.load_state_dict(torch.load("rl_agent/pendulum_policy.pt"))


def rl_policy(_t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A trained RL-based policy.

    Args:
        _t (float | torch.Tensor): time
        x (torch.Tensor): state

    Returns:
        torch.Tensor: control (a vector of zeros)
    """
    return rl_policy_net(x)


# initialize the controlled SDE
# In this example, there is no extra noise
sde = controlled_sde.InvertedPendulum(rl_policy)

# Initialize the batch of starting states
x0 = torch.tensor([STARTING_SPEED, STARTING_ANGLE],
                  device=device).expand(BATCH_SIZE, -1)
ts = torch.linspace(0, DURATION, T_SIZE, device=device)

sample_paths = sde.sample(x0, ts, method="srk").squeeze()
plot_data = sample_paths.cpu().numpy()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_offset(torch.pi/2)
ax.plot(plot_data[:, :, 1], plot_data[:, :, 0])
plt.show()

sde.render(sample_paths, ts)
sde.close()
