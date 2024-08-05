import torch
from torch.nn import Module
import tqdm
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from controlled_sde import ControlledSDE
from .sampling import Sampler
from .specification import Specification
from .membership_sets import SublevelSet, difference


class SupermartingaleCertificate():
    def __init__(self,
                 sde: ControlledSDE,
                 specification: Specification,
                 sampler: Sampler,
                 net: Module,
                 device: torch.device
                 ):
        super().__init__()
        self.sde = sde
        self.specification = specification
        self.sampler = sampler
        self.net = net
        self.device = device

    def train(self,
              n_epochs: int = 10_000,
              dt: float = 0.05,
              n_time: int | None = None,
              n_space: int = 4096,
              batch_size: int = 256,
              lr: float = 1e-3,
              zeta: float = 0.1,
              xi: float = 0.1
              ):
        # initialize auxiliary variables
        spec = self.specification
        V = self.net
        V.train(True)
        generator = self.sde.generator(V, spec.time_homogenous)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        sampler = self.sampler

        # sample points from the time-state space
        x = torch.tensor(sampler.sample_space(n_space),
                         dtype=torch.float32,
                         device=self.device
                         )

        if not spec.time_homogenous:
            t = torch.tensor(sampler.sample_time(
                n_time, dt), dtype=torch.float32)
            t = torch.repeat_interleave(t, n_space)
            x = torch.tile(x, (n_time, 1))
            x = torch.cat((t.view(-1, 1), x), dim=1)

        x_star = torch.zeros((1, x.shape[1]))

        # compute the threshold constants
        prob_ra = spec.reach_avoid_probability
        prob_stay = spec.stay_probability

        alpha_ra = 1.0
        beta_ra = alpha_ra / (1.0 - prob_ra)
        beta_s = alpha_ra * 0.1
        beta_s_inner = beta_s
        alpha_s = beta_s * (1.0 - prob_stay)

        # initialize the sets
        target_interior = spec.target_set.interior
        non_target_area = difference(
            SublevelSet(V, beta_ra),
            target_interior
        )
        outer_area = difference(
            SublevelSet(V, beta_s),
            target_interior
        )
        sub_alpha_set = SublevelSet(V, alpha_s)

        # run training for n_epochs
        progress_bar = tqdm.tqdm(range(n_epochs))
        for _ in progress_bar:
            # reset the gradients
            optimizer.zero_grad()
            x.requires_grad = False

            # sample a batch of points from the grid
            indices = torch.randint(n_space, (batch_size, )).to(self.device)
            batch = x[indices, :]

            # find the loss over the initial set points
            x_0 = spec.initial_set.filter(batch)
            if torch.numel(x_0) != 0:
                init_loss = torch.clamp(V(x_0) - alpha_ra, min=0.0).sum()
            else:
                init_loss = 0.0

            # find the loss over the safety set points
            x_u = spec.unsafe_set.filter(batch)
            if torch.numel(x_u) != 0:
                safety_loss = torch.clamp(beta_ra - V(x_u), min=0.0).sum()
            else:
                safety_loss = 0.0

            # find the loss over the interior of the target set
            x_inner = target_interior.filter(batch)
            goal_loss_in = V(x_star).item()
            if torch.numel(x_inner) != 0:
                values = V(x_inner)
                goal_loss_in += torch.clamp(
                    values - beta_s, min=0.0
                ).sum()
                # use the largest value of V(x) inside the target set
                # as a cutoff point for the stay condition
                beta_s_inner = values.detach().max().cpu().item()

            # find the loss outside of the target set
            x_outer = outer_area.filter(batch)
            # print(x_outer, V(x_outer))
            if torch.numel(x_outer) != 0:
                goal_loss_out = torch.clamp(
                    beta_s - V(x_outer), min=0.0
                ).sum()
            else:
                goal_loss_out = 0.0

            # find the loss for the infinitesimal generator for reach-avoid
            x_non_target = non_target_area.filter(batch)
            if torch.numel(x_non_target) != 0:
                gen_values = generator(x_non_target)
                decrease_loss = torch.clamp(
                    gen_values, min=-zeta).sum()
            else:
                decrease_loss = 0.0

            # find the loss for the infinitesimal generator for stay
            alpha_s = beta_s_inner * (1.0 - prob_stay)
            sub_alpha_set.threshold = alpha_s
            x_stay = sub_alpha_set.complement.filter(x_inner)
            if torch.numel(x_stay) != 0:
                gen_values_stay = generator(x_stay)
                stay_loss = torch.clamp(
                    gen_values_stay, min=-xi).sum()
            else:
                stay_loss = 0.0

            # find the total loss
            loss = init_loss + safety_loss + \
                goal_loss_in + goal_loss_out + \
                decrease_loss + stay_loss

            # update the progress bar with the loss information
            progress_bar.set_description(
                f"L0:{init_loss: 8.3f}, "
                f"Lu:{safety_loss: 8.3f}, "
                f"Lgi:{goal_loss_in: 8.3f}, "
                f"Lgo:{goal_loss_out: 8.3f}, "
                f"Ld:{decrease_loss: 8.3f}, "
                f"Ls:{stay_loss: 8.3f}"
            )

            # if the loss is scalar (i.e., a batch sample was not
            # representative), go to the next step
            if isinstance(loss, float):
                continue

            # do the gradient step
            torch.nn.utils.clip_grad_norm_(V.parameters(), 1.0)
            loss.backward()
            optimizer.step()

    def verify(self):
        # reach-avoid probability
        x0 = torch.tensor([[0.0, torch.pi]],
                          dtype=torch.float32,
                          device=self.device
                          )
        x0_lb = torch.tensor([[-0.5, torch.pi*7/8]],
                             dtype=torch.float32,
                             device=self.device
                             )
        x0_ub = torch.tensor([[+0.5, torch.pi]],
                             dtype=torch.float32,
                             device=self.device
                             )
        lirpa_model = BoundedModule(self.net, torch.empty_like(
            x0), bound_opts={'sparse_intermediate_bounds': False})
        norm = float("inf")
        ptb_x0 = PerturbationLpNorm(norm=norm, x_L=x0_lb, x_U=x0_ub)
        bounded_x0 = BoundedTensor(x0, ptb_x0)
        _, ub = lirpa_model.compute_bounds(
            x=(bounded_x0,), method='alpha-CROWN')
        init_upper = ub.item()
        xu = torch.tensor([[7.0, torch.pi/2.0]],
                          dtype=torch.float32,
                          device=self.device
                          )
        xu_lb = torch.tensor([[6.0, 0.0]],
                             dtype=torch.float32,
                             device=self.device
                             )
        xu_ub = torch.tensor([[8.0, torch.pi]],
                             dtype=torch.float32,
                             device=self.device
                             )
        ptb_xu = PerturbationLpNorm(norm=norm, x_L=xu_lb, x_U=xu_ub)
        bounded_xu = BoundedTensor(xu, ptb_xu)
        lb, _ = lirpa_model.compute_bounds(
            x=(bounded_xu,), method='alpha-CROWN')
        unsafe_lower = lb.item()
        xu = torch.tensor([[-7.0, -torch.pi/2.0]],
                          dtype=torch.float32,
                          device=self.device
                          )
        xu_lb = torch.tensor([[-8.0, -torch.pi]],
                             dtype=torch.float32,
                             device=self.device
                             )
        xu_ub = torch.tensor([[-6.0, 0.0]],
                             dtype=torch.float32,
                             device=self.device
                             )
        ptb_xu = PerturbationLpNorm(norm=norm, x_L=xu_lb, x_U=xu_ub)
        bounded_xu = BoundedTensor(xu, ptb_xu)
        lb, _ = lirpa_model.compute_bounds(
            x=(bounded_xu,), method='alpha-CROWN')
        if unsafe_lower > lb.item():
            unsafe_lower = lb.item()
        print(f"max initial value: {init_upper}")
        print(f"min unsafe value: {unsafe_lower}")
        prob_ra = max(1 - init_upper/unsafe_lower, 0)
        print(
            f"reach-avoid constraint satisfied with "
            f"probability at least {prob_ra}"
        )
        if prob_ra < self.specification.reach_avoid_probability:
            return False
        # # decrease property
        # beta_ra = init_upper / \
        #     (1.0 - self.specification.reach_avoid_probability)
        # n_samples = 4096
        # # min distance between sampled points for Sobol sequences is tol,
        # # there is probably a better number to use here
        # tol = 0.5 * (2 ** 0.5) / n_samples
        # sublevel_sample = SublevelSet(self.net, beta_ra).filter(
        #     torch.tensor(self.sampler.sample_space(4096),
        #                  dtype=torch.float32,
        #                  device=self.device
        #                  )
        # )
        # sublevel_lb = torch.min(sublevel_sample, dim=0,
        #                         keepdim=True).values - tol
        # sublevel_ub = torch.max(sublevel_sample, dim=0,
        #                         keepdim=True).values + tol
        # x = torch.tensor([[0.0, 0.0]],
        #                  dtype=torch.float32,
        #                  device=self.device
        #                  )
        # ptb = PerturbationLpNorm(norm=norm, x_L=sublevel_lb, x_U=sublevel_ub)
        # bounded_x = BoundedTensor(x, ptb)
        # generator = self.sde.generator(self.net, True)
        # lirpa_model = BoundedModule(generator, torch.empty_like(x0))
        # # the line above will throw because
        # # generator is not a module but a function
        # lb, ub = lirpa_model.compute_bounds(
        #     x=(bounded_x,), method='alpha-CROWN')
        return True
