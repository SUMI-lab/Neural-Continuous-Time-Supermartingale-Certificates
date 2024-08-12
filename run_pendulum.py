import torch
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import controlled_sde
from rl_agent import TanhPolicy
import stochastic_rsa as rsa

# Seed the random number generators
seeds = npr.randint(1, 1e5, size=(2,))
torch.manual_seed(0)
npr.seed(seeds[1])
torch.set_default_dtype(torch.float32)
torch.use_deterministic_algorithms(True)


DEVICE_STR = "cpu"           # Torch device

STARTING_ANGLE = torch.pi    # starting angle for plotting sample paths
STARTING_SPEED = 0.0         # starting angular velocity
DURATION = 60                # seconds
FPS = 20                     # frames per second
T_SIZE = DURATION * FPS + 1  # number of time steps for each sample path

# Initialize the device for torch
if DEVICE_STR == "auto":
    if torch.cuda.is_available() and torch.backends.cuda.is_built():
        DEVICE_STR = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE_STR = "mps"
    else:
        DEVICE_STR = "cpu"
device = torch.device(DEVICE_STR)

# load the policy
rl_policy_net = TanhPolicy(2, 1, 64, device=device)
rl_policy_net.load_state_dict(torch.load(
    "rl_agent/pendulum_policy.pt",
    map_location=device,
    weights_only=True
))
rl_policy_net.requires_grad_(False)

# initialize the controlled SDE
sde = controlled_sde.InvertedPendulum(rl_policy_net)
net = rsa.CertificateModule(device=device)

# set the boundaries of the sets
global_bounds = np.array([[[-20.0, -2*np.pi], [20.0, 2*np.pi]]])
initial_bounds = np.array([[[-0.5, 7/8*np.pi], [0.5, 9/8*np.pi]]])
target_bounds = np.array([[[-4.0, -np.pi/2], [4.0, np.pi/2]]])
unsafe_bounds = np.array([
    [[-20.0, -2*np.pi], [-10.0, -3/2*np.pi]],
    [[10.0, 3/2*np.pi], [20.0, 2*np.pi]]
])

# create the sets
interest_set = rsa.AABBSet(global_bounds, device)
initial_set = rsa.AABBSet(initial_bounds, device)
target_set = rsa.AABBSet(target_bounds, device)
unsafe_set = rsa.AABBSet(unsafe_bounds, device)
reach_avoid_probability, stay_probability = 0.9, 0.9

# create the specification
spec = rsa.Specification(
    interest_set,
    initial_set,
    unsafe_set,
    target_set,
    0.9,
    0.9
)

# create the certificate
certificate = rsa.SupermartingaleCertificate(sde, spec, net, device)

# train the certificate
certificate.train(n_epochs=10000, batch_size=64, lr=1e-3,
                  verify_every_n=1000, verifier_mesh_size=400, zeta=1.0)

# Initialize the batch of starting states
x0 = torch.tile(torch.tensor([[STARTING_SPEED, STARTING_ANGLE]],
                             device=device), dims=(4, 1))
ts = torch.linspace(0, 0.1*DURATION, T_SIZE, device=device)

#
sample_paths = sde.sample(x0, ts, method="srk").squeeze()

# Plot
fig, ax1 = plt.subplots(1, 1)

with torch.no_grad():
    print(global_bounds[0, 0, 0])
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(global_bounds[0, 0, 0],
                           global_bounds[0, 1, 0], 101),
            torch.linspace(global_bounds[0, 0, 1],
                           global_bounds[0, 1, 1], 101),
            indexing='xy'
        )
    )
    grid = grid.reshape(2, -1).T
    out = certificate.net(grid).detach().numpy().reshape(101, 101)
    # scaling_factor = certificate.net(
    #     initial_set.sample(1000)
    # ).detach().numpy().max()
    # out /= scaling_factor
    min_level = int(np.floor(np.log10(out.min()) * 5))
    max_level = int(np.ceil(np.log10(out.max()) * 5)) + 1
    c = ax1.contourf(
        np.linspace(global_bounds[0, 0, 0], global_bounds[0, 1, 0], 101),
        np.linspace(global_bounds[0, 0, 1], global_bounds[0, 1, 1], 101),
        out,
        norm=colors.LogNorm(),
        levels=[10 ** (n / 5) for n in range(min_level, max_level, 1)]
    )

fig.colorbar(c, ax=ax1)

ax1.set_xlim(global_bounds[0, :, 0])
ax1.set_ylim(global_bounds[0, :, 1])
for i in range(initial_bounds.shape[0]):
    ax1.add_patch(Rectangle(initial_bounds[i, 0, :], *(initial_bounds[i, 1, :] - initial_bounds[i, 0, :]),
                            edgecolor='yellow',
                            facecolor='none',
                            lw=2))
for i in range(target_bounds.shape[0]):
    ax1.add_patch(Rectangle(target_bounds[i, 0, :], *(target_bounds[i, 1, :] - target_bounds[i, 0, :]),
                            edgecolor='green',
                            facecolor='none',
                            lw=2))
for i in range(unsafe_bounds.shape[0]):
    ax1.add_patch(Rectangle(unsafe_bounds[i, 0, :], *(unsafe_bounds[i, 1, :] - unsafe_bounds[i, 0, :]),
                            edgecolor='red',
                            facecolor='none',
                            lw=2))

path_data = sample_paths.numpy()
ax1.plot(path_data[:, :, 0], path_data[:, :, 1],
         color="white", lw=1, alpha=0.5
         )

plt.show()

# sde.render(sample_paths, ts)
sde.close()
