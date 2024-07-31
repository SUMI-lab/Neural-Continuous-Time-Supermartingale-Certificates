import controlled_sde
import torch
import numpy.random as npr
import matplotlib.pyplot as plt

seeds = npr.randint(1, 1e5, size=(2,))
torch.manual_seed(seeds[0])
npr.seed(seeds[1])

device = "cpu"   # Torch device
batch_size = 64  # how many environments to run in parallel
duration = 5     # seconds
fps = 60         # frames per second

starting_angle = torch.pi/8  # starting angle for each sample path
starting_speed = 0.0         # starting angular velocity
t_size = duration * fps + 1  # number of time steps for each sample path


if device == "auto":
    if torch.cuda.is_available() and torch.backends.cuda.is_built():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

device = torch.device(device)


def policy_do_nothing(t, x):
    # this policy is always zero and does nothing
    return torch.zeros((x.size(0),), device=device).unsqueeze(1)


# initialize the controlled SDE
# In this example, there is no extra noise
sde = controlled_sde.InvertedPendulum(policy_do_nothing)

# Initialize the batch of starting states
x0 = torch.tensor([starting_speed, starting_angle],
                  device=device).expand(batch_size, -1)
ts = torch.linspace(0, duration, t_size, device=device)

sample_paths = sde.sample(x0, ts, method="srk").squeeze()
plot_data = sample_paths.cpu().numpy()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_offset(torch.pi/2)
ax.plot(plot_data[:, :, 1], plot_data[:, :, 0])
plt.show()

sde.render(sample_paths, ts)
