import controlled_sde
import torch
import numpy.random as npr
import matplotlib.pyplot as plt

seeds = npr.randint(1, 1e5, size=(2,))
torch.manual_seed(seeds[0])
npr.seed(seeds[1])

mps_device = torch.device("mps")

batch_size = 16  # how many environments to run in parallel
duration = 5     # seconds
fps = 60         # frames per second

starting_angle = torch.pi/8  # starting angle for each sample path
starting_speed = 0.0         # starting angular velocity
t_size = duration * fps + 1  # number of time steps for each sample path


def policy_do_nothing(t, x):
    # this policy is always zero and does nothing
    return torch.zeros((x.size(0),)).unsqueeze(1)


# initialize the controlled SDE
# In this example, there is no extra noise
sde = controlled_sde.InvertedPendulum(
    policy_do_nothing, volatility_scale=2.0)

# Initialize the batch of starting states
x0 = torch.Tensor([starting_speed, starting_angle]).expand(batch_size, -1)
ts = torch.linspace(0, duration, t_size)

solution = sde.sample(x0, ts, method="srk").squeeze()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_offset(torch.pi/2)
ax.plot(solution.numpy()[:, :, 1], solution.numpy()[:, :, 0])
plt.show()

sde.render(solution, ts)
