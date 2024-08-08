from typing import Sequence
import torch
from torch.nn import Module
import tqdm
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.operators.jacobian import JacobianOP, GradNorm
from controlled_sde import ControlledSDE
from .sampling.grid import GridSampler
from .sampling import Sampler
from .specification import Specification
from .membership_sets import SublevelSet, difference, intersection
from .net import CertificateNet


def lerp(x1: float, x2: float, rate: float):
    rate = max(min(rate, 1.0), 0.0)  # clamp to [0, 1]
    return x1 * (1.0 - rate) + x2 * rate


# class JacobianModule(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, x):
#         y = self.model(x)
#         return JacobianOP.apply(y, x)


class generatorModule(torch.nn.Module):
    def __init__(self, model: CertificateNet, sde: ControlledSDE):
        super().__init__()
        self.model = model
        self.policy = sde.policy
        self.sde = sde

    def forward(self, x, dvdx, d2v_dx2):
        f = self.sde.generator(x, dvdx, d2v_dx2)
        return f


class SupermartingaleCertificate():
    def __init__(self,
                 sde: ControlledSDE,
                 specification: Specification,
                 sampler: Sampler,
                 net: CertificateNet,
                 device: torch.device
                 ):
        super().__init__()
        self.sde = sde
        self.specification = specification
        self.sampler = sampler
        self.net = net
        self.device = device
        dummy_x = torch.tensor(sampler.sample_space(100),
                               dtype=torch.float32,
                               device=self.device
                               )
        dummy_d = torch.ones_like(dummy_x)
        dummy_d2 = torch.zeros_like(dummy_x)
        self.level_verifier = BoundedModule(
            self.net,
            (dummy_x, dummy_d),
            device=self.device
        )
        self.decrease_verifier = BoundedModule(
            generatorModule(self.net, self.sde),
            (dummy_x, dummy_d, dummy_d2),
            device=self.device
        )
        # self.jacobian_module = BoundedModule(
        #     JacobianModule(self.net),
        #     (dummy_x,),
        #     device=self.device
        # )

    def train(self,
              n_epochs: int = 10_000,
              dt: float = 0.05,
              n_time: int | None = None,
              n_space: int = 4096,
              batch_size: int = 256,
              lr: float = 1e-3,
              zeta: float = 1e-1,
              verify_every_n: int = 1000,
              annealing: float = 0.5,
              regularizer_lambda: float = 0.001,
              verifier_mesh_size: int = 41
              ):
        # initialize auxiliary variables
        spec = self.specification
        V = self.net
        V.train(True)
        generator = self.sde.generator
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

        n_dim = x.shape[1]
        x_star = torch.zeros((1, n_dim), device=self.device)
        decrease_counterexamples = torch.zeros((1, n_dim), device=self.device)
        x_outer_counterexamples = torch.empty((0, n_dim), device=self.device)

        # compute the threshold constants
        prob_ra = spec.reach_avoid_probability
        prob_stay = spec.stay_probability

        # initialize the sets
        target_interior = spec.target_set.interior

        sub_beta_ra_set = SublevelSet(V, 1.0)
        non_target_area = difference(
            sub_beta_ra_set,
            target_interior
        )
        sub_alpha_ra_set = SublevelSet(
            V,
            sub_beta_ra_set.threshold * (1.0 - prob_ra)
        )

        sub_beta_s_set = SublevelSet(
            V, sub_alpha_ra_set.threshold * 0.1)
        outer_area = difference(
            sub_beta_s_set,
            target_interior
        )
        sub_alpha_s_set = SublevelSet(
            V,
            sub_beta_s_set.threshold * (1.0 - prob_stay)
        )
        beta_ra_alpha_s_band = difference(
            sub_beta_ra_set,
            sub_alpha_s_set
        )

        # initialize verifier
        if isinstance(sampler, GridSampler):
            mesh_sampler = sampler
        else:
            mesh_sampler = GridSampler(
                low=sampler.low,
                high=sampler.high
            )
        cells = torch.tensor(
            mesh_sampler.sample_cells(verifier_mesh_size ** n_dim),
            dtype=torch.float32,
            device=self.device
        )

        # run training for n_epochs
        progress_bar = tqdm.tqdm(range(n_epochs))
        for epoch in progress_bar:
            # reset the gradients
            optimizer.zero_grad()
            x.requires_grad = False

            # sample a batch of points from the grid
            indices = torch.randint(n_space, (batch_size, )).to(self.device)
            batch = x[indices, :]

            # find the loss over the initial set points
            x_0 = spec.initial_set.filter(x)
            if torch.numel(x_0) != 0:
                init_loss = torch.clamp(
                    V(x_0) - sub_alpha_ra_set.threshold,
                    min=0.0
                ).sum()
            else:
                init_loss = 0.0

            # find the loss over the safety set points
            x_u = spec.unsafe_set.filter(x)
            if torch.numel(x_u) != 0:
                safety_loss = torch.clamp(
                    sub_beta_ra_set.threshold - V(x_u),
                    min=0.0
                ).sum()
            else:
                safety_loss = 0.0

            # find the loss over the interior of the target set
            # if x_inner_counterexamples.shape[0] > batch_size:
            #     indices = torch.randint(
            #         x_inner_counterexamples.shape[0],
            #         (batch_size, )
            #     ).to(self.device)
            #     inner_counterexample_batch = x_inner_counterexamples[indices, :]
            # else:
            #     inner_counterexample_batch = x_inner_counterexamples
            # x_inner = torch.cat(
            #     (target_interior.filter(batch), inner_counterexample_batch),
            #     dim=0
            # )
            # if torch.numel(x_inner) != 0:
            #     values = V(x_inner)
            #     goal_loss_in = torch.clamp(
            #         values - sub_beta_s_set.threshold,
            #         min=0.0
            #     ).sum()
            # else:
            #     goal_loss_in = 0.0

            # find the loss outside of the target set
            if x_outer_counterexamples.shape[0] > batch_size:
                indices = torch.randint(
                    x_outer_counterexamples.shape[0],
                    (batch_size, )
                ).to(self.device)
                outer_counterexample_batch = x_outer_counterexamples[indices, :]
            else:
                outer_counterexample_batch = x_outer_counterexamples
            x_outer = torch.cat(
                (outer_area.filter(batch), outer_counterexample_batch),
                dim=0
            )
            # print(x_outer, V(x_outer))
            if torch.numel(x_outer) != 0:
                goal_loss_out = torch.clamp(
                    sub_beta_s_set.threshold -
                    V(x_outer),
                    min=0.0
                ).sum()
            else:
                goal_loss_out = 0.0

            # find the loss for the infinitesimal generator for reach-avoid
            if decrease_counterexamples.shape[0] > batch_size:
                indices = torch.randint(
                    decrease_counterexamples.shape[0],
                    (batch_size, )
                ).to(self.device)
                decrease_counterexample_batch = decrease_counterexamples[indices, :]
            else:
                decrease_counterexample_batch = decrease_counterexamples
            x_decrease = torch.cat(
                (
                    beta_ra_alpha_s_band.filter(batch),
                    decrease_counterexample_batch
                ),
                dim=0
            )
            if torch.numel(x_decrease) != 0:
                # n_points = x_decrease.shape[0]
                # f_value = self.sde.f(x_decrease)
                # g_value = self.sde.g(x_decrease)
                # _, vjpfunc = torch.func.vjp(V, x_decrease)
                # vjps = vjpfunc(torch.ones(
                #     (n_points, 1), device=x_decrease.device))
                # nabla = vjps[0]
                # _, vjpfunc2 = torch.func.vjp(
                #     lambda y: vjpfunc(torch.ones((n_points, 1), device=y.device))[0], x_decrease)
                # vjps2 = vjpfunc2(torch.ones(
                #     (n_points, 1), device=x_decrease.device))
                # hessian_diag = vjps2[0]
                # print(g_value.shape, hessian_diag.shape)
                # gen_values = (f_value * nabla).sum(dim=1) + 0.5 * \
                #     (torch.square(g_value) * hessian_diag).sum(dim=1)

                _, dv_dx, d2v_dx2 = V(
                    x_decrease,
                    return_derivatives=True
                )
                gen_values = generator(x_decrease, dv_dx, d2v_dx2)
                decrease_loss = torch.clamp(
                    gen_values + zeta, min=0.0).sum()
            else:
                decrease_loss = 0.0

            # find the loss regularizer
            regularizer = regularizer_lambda
            for layer in V.children():
                if isinstance(layer, torch.nn.Linear):
                    regularizer *= torch.linalg.matrix_norm(
                        layer.weight,
                        ord=torch.inf
                    )

            # find the total loss
            loss = init_loss + safety_loss + \
                goal_loss_out + \
                decrease_loss + \
                V(x_star) + regularizer

            # update the progress bar with the loss information
            progress_bar.set_description(
                f"L0:{init_loss: 6.3f}, "
                f"Lu:{safety_loss: 8.3f}, "
                f"Lg:{goal_loss_out: 6.3f}, "
                f"Ld:{decrease_loss: 8.3f}, "
                f"reg:{regularizer: 6.3f}"
            )

            # if the loss is scalar (i.e., a batch sample was not
            # representative), go to the next step
            if isinstance(loss, float):
                continue

            # do the gradient step
            if False and (epoch + 1) % verify_every_n == 0 or epoch + 1 == n_epochs:
                # verify
                print("Verifying:")

                # Verification step 1. Find sublevel set covers with perturbation analysis
                cell_lb, cell_ub = self._bound_estimate(
                    cells,
                    self.level_verifier,
                    mesh_size=verifier_mesh_size
                )

                # Verification step 2. reach-avoid probability
                # first, upper bound the certificate value at the initial states
                init_upper = torch.max(
                    cell_ub[spec.initial_set.contains(cells), :]
                ).item()
                # init_upper = self._bound_estimate(
                #     [[0.0, torch.pi*15/16]],
                #     [[-0.5, torch.pi*7/8]],
                #     [[+0.5, torch.pi]],
                #     bound_lower=False,
                #     method="alpha-CROWN"
                # )[1].item()

                # # next, lower bound the certificate value at the unsafe states
                # unsafe_lower_1 = self._bound_estimate(
                #     [[7.0, torch.pi/2.0]],
                #     [[6.0, 0.0]],
                #     [[8.0, torch.pi]],
                #     bound_upper=False,
                #     method="alpha-CROWN"
                # )[0].item()
                # unsafe_lower_2 = self._bound_estimate(
                #     [[-7.0, -torch.pi/2.0]],
                #     [[-8.0, -torch.pi]],
                #     [[-6.0, 0.0]],
                #     bound_upper=False,
                #     method="alpha-CROWN"
                # )[0].item()
                # unsafe_lower = min(unsafe_lower_1, unsafe_lower_2)
                unsafe_lower = torch.min(
                    cell_lb[spec.unsafe_set.contains(cells), :]
                ).item()
                # compute the estimated reach-avoid probability
                # print(init_upper, unsafe_lower)
                if unsafe_lower < 0.0:
                    unsafe_lower = 0.0
                    prob_ra_estimate = 0.0
                else:
                    prob_ra_estimate = max(1.0 - init_upper/unsafe_lower, 0.0)
                print(
                    f"Reach-avoid condition is satisfied with "
                    f"probability at least {prob_ra_estimate}."
                )

                # If it is higher than the specified probability, we can lower
                # the bounds. This helps with keeping the sub-sub_beta_ra_set.threshold set small
                # speeding up the computations. We changes the values slowly
                # so that the probability stays high; in particular, by changing
                # beta based on the previous alpha values, we make sure it is
                # only going down in a safer manner.
                prob_threshold = (1.0 - prob_ra) * annealing
                if 1.0 - prob_ra_estimate < prob_threshold:
                    sub_beta_ra_set.threshold = min(
                        sub_beta_ra_set.threshold,
                        sub_alpha_ra_set.threshold / (1.0 - prob_ra)
                    )
                    sub_alpha_ra_set.threshold = lerp(
                        sub_alpha_ra_set.threshold, max(
                            init_upper,
                            sub_beta_s_set.threshold
                        ),
                        0.5
                    )
                elif prob_ra_estimate < prob_ra:
                    sub_beta_ra_set.threshold = max(
                        sub_beta_ra_set.threshold,
                        sub_alpha_ra_set.threshold / (1.0 - prob_ra)
                    )

                # Verification step 3. stay probability
                # first, find a new alpha_s level if needed; we use the largest
                # upper bound of a cell, as all of the cell is then within this
                # sublevel set
                beta_s_candidate_in = torch.max(
                    cell_ub[spec.target_set.contains(cells), :]
                ).item()
                beta_s_candidate_out = torch.max(
                    cell_lb[spec.target_set.complement.contains(cells), :]
                ).item()
                # target_upper = self._bound_estimate(
                #     [[0.0, 0.0]],
                #     [[-2.0, -torch.pi/3.0]],
                #     [[2.0, torch.pi/3.0]],
                #     bound_lower=False,
                #     method="alpha-CROWN"
                # )[1].item()
                # print(beta_s_candidate_in, beta_s_candidate_out, target_upper)
                beta_s_candidate = min(
                    beta_s_candidate_in, beta_s_candidate_out)
                if beta_s_candidate < sub_alpha_ra_set.threshold:
                    sub_beta_s_set.threshold = lerp(
                        beta_s_candidate,
                        sub_beta_s_set.threshold,
                        0.5
                    )
                    sub_alpha_s_set.threshold = sub_beta_s_set.threshold * (
                        1.0 - prob_stay)

                alpha_s_candidate = torch.min(
                    cell_ub[spec.target_set.contains(cells), :]
                ).item()
                prob_s_estimate = max(
                    1.0 - alpha_s_candidate/beta_s_candidate, 0.0)
                print(
                    f"Stay condition is satisfied with "
                    f"probability at least {prob_s_estimate}."
                )

                mask = torch.logical_and(
                    spec.target_set.complement.contains(cells),
                    (cell_lb <= sub_beta_s_set.threshold).squeeze()
                )
                x_outer_counterexamples = cells[mask, :]
                n_counterexamples = x_outer_counterexamples.shape[0]
                if n_counterexamples > 0:
                    print(
                        f"Found {n_counterexamples} potential "
                        "goal condition violations outside of the target set."
                    )

                # Verification step 4. decrease condition
                mask = torch.logical_and(
                    cell_lb > sub_alpha_s_set.threshold,
                    cell_ub <= sub_beta_ra_set.threshold
                ).squeeze()
                decrease_cells = cells[mask, :]
                # print(sub_alpha_s_set.threshold, sub_beta_ra_set.threshold)
                dc = decrease_cells
                if torch.numel(dc) > 0:
                    bounded_decrease_cells = self._to_bounded_tensor(
                        dc,
                        mesh_size=verifier_mesh_size
                    )
                    # lb_jac, ub_jac = self.jacobian_module.compute_jacobian_bounds(
                    #     x=(bounded_decrease_cells,)
                    # )
                    # print(lb_jac, ub_jac)
                    # bounded_gradient_cells = self._to_bounded_tensor(
                    #     self.jacobian_module(decrease_cells),
                    #     perturbation_lower=lb_jac,
                    #     perturbation_upper=ub_jac
                    # )
                    df = BoundedTensor(
                        torch.ones_like(dc),
                        PerturbationLpNorm(eps=0)
                    )
                    df2 = BoundedTensor(
                        torch.zeros_like(dc),
                        PerturbationLpNorm(eps=0)
                    )
                    _, ub = self.decrease_verifier.compute_bounds(
                        x=(
                            bounded_decrease_cells,
                            df,
                            df2
                        )
                    )
                    # print(ub)
                    mask = ub > -zeta
                    decrease_counterexamples = decrease_cells[mask, :]
                    if decrease_counterexamples.shape[0] > 0:
                        zeta = torch.max(ub).item()
                        print(
                            f"Found {decrease_counterexamples.shape[0]} "
                            "potential decrease condition violations. "
                            f"Maximum value is {zeta}"
                        )

                print(
                    f"beta_ra is {sub_beta_ra_set.threshold}, "
                    f"alpha_ra is {sub_alpha_ra_set.threshold}, "
                    f"beta_s is {sub_beta_s_set.threshold}, "
                    f"alpha_s is {sub_alpha_s_set.threshold}."
                )

            # do the gradient step
            torch.nn.utils.clip_grad_norm_(V.parameters(), 1.0)
            loss.backward()
            optimizer.step()

    def _to_bounded_tensor(
        self,
            values: Sequence[Sequence[float]],
            perturbation_lower: Sequence[Sequence[float]] | None = None,
            perturbation_upper: Sequence[Sequence[float]] | None = None,
            mesh_size: int | None = None
    ):
        if isinstance(values, torch.Tensor):
            x = values
        else:
            x = torch.tensor(values,
                             dtype=torch.float32,
                             device=self.device
                             )
        if perturbation_lower is None or perturbation_upper is None:
            if mesh_size is None:
                raise ValueError("No perturbation bounds or mesh size given.")
            magnitude = torch.tensor(self.sampler.magnitude,
                                     dtype=torch.float32,
                                     device=self.device
                                     )
            eps = magnitude / (mesh_size - 1)
            x_lb, x_ub = x - eps, x + eps
        else:
            if not isinstance(perturbation_lower, torch.Tensor):
                x_lb = torch.tensor(perturbation_lower,
                                    dtype=torch.float32,
                                    device=self.device
                                    )
            else:
                x_lb = perturbation_lower.to(self.device)
            if not isinstance(perturbation_upper, torch.Tensor):
                x_ub = torch.tensor(perturbation_upper,
                                    dtype=torch.float32,
                                    device=self.device
                                    )
            else:
                x_ub = perturbation_upper.to(self.device)
        perturbation = PerturbationLpNorm(
            norm=float("inf"),
            x_L=x_lb,
            x_U=x_ub
        )
        bounded_x = BoundedTensor(x, perturbation)
        return bounded_x

    def _bound_estimate(
            self,
            values: Sequence[Sequence[float]],
            verifier,
            perturbation_lower: Sequence[Sequence[float]] | None = None,
            perturbation_upper: Sequence[Sequence[float]] | None = None,
            mesh_size: int | None = None,
            bound_lower: bool = True,
            bound_upper: bool = True,
            method: str = "backward"
    ):
        bounded_x = self._to_bounded_tensor(
            values,
            perturbation_lower,
            perturbation_upper,
            mesh_size
        )
        lb, ub = verifier.compute_bounds(
            x=(bounded_x,),
            method=method,
            bound_lower=bound_lower,
            bound_upper=bound_upper
        )
        return lb, ub
