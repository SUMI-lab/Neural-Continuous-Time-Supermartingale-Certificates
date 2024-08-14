import csv
import time
import torch
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import controlled_sde
import stochastic_rsa as rsa

torch.set_default_dtype(torch.float32)
torch.use_deterministic_algorithms(True)

device = torch.device("cpu")

# initialize the controlled SDE
sde = controlled_sde.GBM()
net = rsa.CertificateModule(device=device)

# set the boundaries of the sets
global_bounds = np.array([[[-100.0, -100.0], [100.0, 100.0]]])
initial_bounds = np.array([[[45, -55], [55, -45]]])
target_bounds = np.array([[[-25.0, -25.0], [25.0, 25.0]]])
unsafe_bounds = np.array([
    [[-100.0, -100.0], [-80.0, 100.0]],
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

npr.seed(1)
seeds = npr.randint(1, 1e5, size=(5,))
for seed in seeds:
    torch.manual_seed(seed)
    npr.seed(seed)

    # create the certificate
    net = rsa.CertificateModule(device=device)
    certificate = rsa.SupermartingaleCertificate(sde, spec, net, device)

    # train the certificate
    t = time.time()
    result = certificate.train(verify_every_n=1000,
                               verifier_mesh_size=200,
                               zeta=1.0,
                               regularizer_lambda=1e-1,
                               verification_slack=4
                               )
    t = time.time() - t
    result = (seed, t, result[0], result[1])
    with open('gbm.csv', 'a') as file:
        writer = csv.writer(file, dialect='excel')
        writer.writerow(result)

# Initialize the batch of starting states
x0 = torch.tile(
    torch.tensor([[50.0, -50.0]], device=device),
    dims=(4, 1)
)
ts = torch.linspace(0, 100.0, 1000, device=device)

#
sample_paths = sde.sample(x0, ts, method="euler").squeeze()

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
    scaling_factor = certificate.net(
        initial_set.sample(1000)
    ).detach().numpy().max()
    out /= scaling_factor
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
                            edgecolor='limegreen',
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
