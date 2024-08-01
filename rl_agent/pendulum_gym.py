from typing import Optional
from collections import defaultdict
from torchrl.envs import (
    CatTensors,
    EnvBase,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torch import nn
from tensordict.nn import TensorDictModule
from tensordict import TensorDict, TensorDictBase
import tqdm
import torch
import numpy as np
from matplotlib import pyplot as plt
from policy import TanhPolicy


DEFAULT_X = np.pi
DEFAULT_Y = 1.0


def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite


class PendulumEnv(EnvBase):
    metadata = {"render_modes": None}
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Helpers: _make_step and gen_params
    @staticmethod
    def gen_params(batch_size=None) -> TensorDictBase:
        """
            Returns a ``tensordict`` containing the physical parameters such as
            gravitational force and torque or speed limits.
        """
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "max_angle": 1.0,
                        "max_speed": 8.0,
                        "max_torque": 6.0,
                        "dt": 0.05,
                        "g": 9.81,
                        "m": 0.15,
                        "l": 0.5,
                        "b": 0.1,
                    },
                    [],
                )
            },
            [],
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    def _make_spec(self, td_params):
        # Under the hood, this will populate self.output_spec["observation"]
        self.observation_spec = CompositeSpec(
            theta=BoundedTensorSpec(
                low=-td_params["params", "max_angle"],
                high=td_params["params", "max_angle"],
                shape=(),
                dtype=torch.float32,
            ),
            phi=BoundedTensorSpec(
                low=-td_params["params", "max_speed"],
                high=td_params["params", "max_speed"],
                shape=(),
                dtype=torch.float32,
            ),
            # we need to add the ``params`` to the observation specs, as we want
            # to pass it at each step during a rollout
            params=make_composite_from_td(td_params["params"]),
            shape=(),
        )
        # since the environment is stateless, we expect the previous output as input.
        # For this, ``EnvBase`` expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()
        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        self.action_spec = BoundedTensorSpec(
            low=-td_params["params", "max_torque"],
            high=td_params["params", "max_torque"],
            shape=(1,),
            dtype=torch.float32,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(*td_params.shape, 1))

    # Mandatory methods: _step, _reset and _set_seed
    def _reset(self, tensordict):
        if tensordict is None or tensordict.is_empty():
            # if no ``tensordict`` is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input ``tensordict`` contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=self.batch_size)

        high_theta = torch.tensor(DEFAULT_X, device=self.device)
        high_phi = torch.tensor(DEFAULT_Y, device=self.device)
        low_theta = -high_theta
        low_phi = -high_phi

        # for non batch-locked environments, the input ``tensordict`` shape dictates the number
        # of simulators run simultaneously. In other contexts, the initial
        # random state's shape will depend upon the environment batch-size instead.
        theta = (
            torch.rand(tensordict.shape, generator=self.rng,
                       device=self.device)
            * (high_theta - low_theta)
            + low_theta
        )
        phi = (
            torch.rand(tensordict.shape, generator=self.rng,
                       device=self.device)
            * (high_phi - low_phi)
            + low_phi
        )
        out = TensorDict(
            {
                "theta": theta,
                "phi": phi,
                "params": tensordict["params"],
            },
            batch_size=tensordict.shape,
        )
        return out

    @staticmethod
    def _step(tensordict):
        theta, phi = tensordict["theta"], tensordict["phi"]  # theta := theta

        g_force = tensordict["params", "g"]
        mass = tensordict["params", "m"]
        length = tensordict["params", "l"]
        dt = tensordict["params", "dt"]
        b = tensordict["params", "b"]
        u = tensordict["action"].squeeze(-1)
        u = tensordict["params", "max_torque"] * u.clamp(-1.0, 1.0)
        costs = angle_normalize(theta) ** 2 + 0.1 * phi**2 + 0.001 * (u**2)

        new_phi = phi + g_force / length * theta.sin() * dt
        new_phi += (u - b * phi) / (mass * length**2) * dt

        new_phi = new_phi.clamp(
            -tensordict["params", "max_speed"],
            tensordict["params", "max_speed"]
        )
        new_theta = theta + new_phi * dt
        reward = -costs.view(*tensordict.shape, 1)
        done = torch.zeros_like(reward, dtype=torch.bool)
        out = TensorDict(
            {
                "theta": new_theta,
                "phi": new_phi,
                "params": tensordict["params"],
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )
        return out

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng


env = PendulumEnv()
env = TransformedEnv(
    env,
    # ``Unsqueeze`` the observations that we will concatenate
    UnsqueezeTransform(
        unsqueeze_dim=-1,
        in_keys=["theta", "phi"],
        in_keys_inv=["theta", "phi"],
    ),
)
cat_transform = CatTensors(
    in_keys=["theta", "phi"], dim=-1, out_key="observation", del_keys=False
)
env.append_transform(cat_transform)

torch.manual_seed(0)
env.set_seed(0)

net = TanhPolicy(1)

policy = TensorDictModule(
    net,
    in_keys=["observation"],
    out_keys=["action"],
)

optim = torch.optim.Adam(policy.parameters(), lr=1e-3)

batch_size = 32
pbar = tqdm.tqdm(range(2_000))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 2_000)
logs = defaultdict(list)

for _ in pbar:
    init_td = env.reset(env.gen_params(batch_size=[batch_size]))
    rollout = env.rollout(100, policy, tensordict=init_td, auto_reset=False)
    traj_return = rollout["next", "reward"].mean()
    (-traj_return).backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optim.step()
    optim.zero_grad()
    pbar.set_description(
        f"reward: {traj_return: 4.4f}, "
        f"last reward: {
            rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}"
    )
    logs["return"].append(traj_return.item())
    logs["last_reward"].append(
        rollout[..., -1]["next", "reward"].mean().item())
    scheduler.step()

torch.save(net.state_dict(), "pendulum_policy.pt")
