import torch
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import controlled_sde
from rl_agent import TanhPolicy
import stochastic_rsa as rsa

# Seed the random number generators
seeds = npr.randint(1, 1e5, size=(2,))
torch.manual_seed(seeds[0])
npr.seed(seeds[1])
torch.set_default_dtype(torch.float32)


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


rl_policy_net = TanhPolicy(1, 64, device=device)
rl_policy_net.load_state_dict(torch.load(
    "rl_agent/pendulum_policy.pt",
    map_location=device,
    weights_only=True
))
rl_policy_net.requires_grad_(False)


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
sde = controlled_sde.InvertedPendulum(rl_policy)

MAX_SPEED = 8.0
MAX_ANGLE = 1.5 * torch.pi

high = torch.tensor([MAX_SPEED, MAX_ANGLE], device=device)
sampler = rsa.sampling.GridSampler(-high.cpu().numpy(),
                                   high.cpu().numpy())
# sampler = rsa.sampling.SobolSampler(-high.cpu().numpy(),
#                                     high.cpu().numpy())
net = rsa.CertificateNet(device=device)

interest_set = rsa.membership_sets.MembershipSet(
    lambda x: torch.all(torch.abs(x) <= high, dim=1)
)

initial_bound = torch.tensor([0.5, torch.pi/8], device=device)
initial_mid = torch.tensor([STARTING_SPEED, STARTING_ANGLE], device=device)
initial_set = rsa.membership_sets.MembershipSet(
    lambda x: torch.all(torch.abs(x - initial_mid) <= initial_bound, dim=1)
)
target_set = rsa.membership_sets.SublevelSet(
    lambda x:
    torch.norm(x / torch.tensor([2.0, torch.pi/0.3],
               device=device), float('inf'), dim=1),
    1.0
)
unsafe_threshold = torch.tensor([6.0, 0.0], device=device)
unsafe_set = rsa.membership_sets.MembershipSet(
    lambda x: torch.all(torch.abs(x) >= unsafe_threshold, dim=1)
)
reach_avoid_probability, stay_probability = 0.9, 0.9

spec = rsa.Specification(
    True,
    interest_set,
    initial_set,
    unsafe_set,
    target_set,
    reach_avoid_probability,
    stay_probability
)

certificate = rsa.SupermartingaleCertificate(sde, spec, sampler, net, device)
certificate.train(n_epochs=10_000, n_space=41*41)
certificate.verify()

# Initialize the batch of starting states
x0 = torch.tensor([STARTING_SPEED, STARTING_ANGLE],
                  device=device).expand(BATCH_SIZE, -1)
ts = torch.linspace(0, DURATION, T_SIZE, device=device)

sample_paths = sde.sample(x0, ts, method="srk").squeeze()

# Plot
fig, ax1 = plt.subplots(1, 1)

with torch.no_grad():
    x = torch.tensor(sampler.sample_space(101*101),
                     dtype=torch.float32,
                     device=device
                     )
    c = ax1.tricontourf(x[:, 0].squeeze().numpy(),
                        x[:, 1].squeeze().numpy(),
                        certificate.net(x).clamp(min=1e-20).squeeze().numpy(),
                        norm=colors.LogNorm())

fig.colorbar(c, ax=ax1)

ax1.set_xlim([-MAX_SPEED, MAX_SPEED])
ax1.set_ylim([-MAX_ANGLE, MAX_ANGLE])
ax1.add_patch(Rectangle((-0.5, 0.875 * torch.pi), 1, 0.25 * torch.pi,
                        edgecolor='yellow',
                        facecolor='none',
                        lw=2))
ax1.add_patch(Rectangle((-2, -torch.pi/3), 4, 2*torch.pi/3,
                        edgecolor='green',
                        facecolor='none',
                        lw=2))
ax1.add_patch(Rectangle((6, 0), 3, 2 * torch.pi,
                        edgecolor='red',
                        facecolor='none',
                        lw=2))
ax1.add_patch(Rectangle((-6, 0), -3, -2 * torch.pi,
                        edgecolor='red',
                        facecolor='none',
                        lw=2))

path_data = sample_paths.cpu().numpy()
ax1.plot(path_data[:, :, 0], path_data[:, :, 1], color="white", lw=1)

plt.show()

sde.render(sample_paths, ts)
sde.close()
