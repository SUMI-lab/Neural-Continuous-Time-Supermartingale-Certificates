from typing import Sequence
import torch
import tqdm
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from controlled_sde import ControlledSDE
from .specification import Specification
from .nets import CertificateModule, GeneratorModule, CertificateModuleWithDerivatives


def lerp(x1: float, x2: float, rate: float):
    rate = max(min(rate, 1.0), 0.0)  # clamp to [0, 1]
    return x1 * (1.0 - rate) + x2 * rate


class SupermartingaleCertificate():
    def __init__(self,
                 sde: ControlledSDE,
                 specification: Specification,
                 net: CertificateModule,
                 device: torch.device
                 ):
        super().__init__()
        # Initialize the parameters
        self.sde = sde
        self.n_dimensions = sde.n_dimensions()
        self.specification = specification
        self.net = net
        self.device = device

        # Create the additional modules for the derivative and generator
        # evaluation
        self.certificate_with_derivatives = CertificateModuleWithDerivatives(
            self.net
        )
        self.generator = GeneratorModule(
            self.certificate_with_derivatives,
            self.sde
        )

        # Create the verification modules
        dummy_x = torch.empty(
            (1, self.n_dimensions),
            dtype=torch.float32,
            device=self.device
        )  # this is required for initialization
        self.level_verifier = BoundedModule(
            self.net,
            (dummy_x,),
            device=self.device
        )  # this module give bounds on the certificate values
        self.decrease_verifier = BoundedModule(
            self.generator,
            (dummy_x,),
            device=self.device
        )  # and this one on the infinitesimal generator values

    def train(self,
              n_epochs: int = 10_000,
              large_sample_size: int = 10000,
              batch_size: int = 256,
              lr: float = 1e-3,
              zeta: float = 1e-2,
              verify_every_n: int = 1000,
              # probability_slack: float = 1.0,
              regularizer_lambda: float = 1e-3,
              decrease_lambda: float = 1.0,
              verifier_mesh_size: int = 400,
              starting_level: float = 1.0
              ):
        # extract the specification information and other useful data
        global_set = self.specification.interest_set
        initial_set = self.specification.initial_set
        unsafe_set = self.specification.unsafe_set
        target_set = self.specification.target_set
        prob_ra = self.specification.reach_avoid_probability
        prob_s = self.specification.stay_probability
        generator = self.generator
        n_dim = self.n_dimensions

        # initialize the certificate training
        v = self.net   # this is the certificate, i.e., V(t, x) in the paper
        v.train(True)  # make sure it's in training mode
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        # compute the threshold constants
        alpha_ra = starting_level
        beta_ra = alpha_ra / (1.0 - prob_ra)
        beta_s = alpha_ra / 2.0
        alpha_s = beta_s * (1.0 - prob_s)

        # x_star = torch.zeros((1, n_dim), device=self.device)

        # initialize the counterexample collections
        decrease_counterexamples = torch.zeros((1, n_dim), device=self.device)
        x_outer_counterexamples = torch.empty((0, n_dim), device=self.device)
        x_inner_counterexamples = torch.empty((0, n_dim), device=self.device)
        n_decrease_counterexamples, n_counterexamples = 0, 0

        cells = torch.meshgrid(
            torch.linspace(global_set.low[0, 0],
                           global_set.high[0, 0], verifier_mesh_size + 1),
            torch.linspace(global_set.low[0, 1],
                           global_set.high[0, 1], verifier_mesh_size + 1),
            indexing='xy'
        )
        # cells = torch.cat(
        #     (torch.reshape(cells[0], (-1,)).unsqueeze(1),
        #      torch.reshape(cells[1], (-1,)).unsqueeze(1)),
        #     dim=1
        # )
        x_L = torch.cat(
            (torch.reshape(cells[0][:-1, :-1], (-1,)).unsqueeze(1),
             torch.reshape(cells[1][:-1, :-1], (-1,)).unsqueeze(1)),
            dim=1
        )
        x_U = torch.cat(
            (torch.reshape(cells[0][1:, 1:], (-1,)).unsqueeze(1),
             torch.reshape(cells[1][1:, 1:], (-1,)).unsqueeze(1)),
            dim=1
        )
        cells = BoundedTensor(
            0.5 * (x_L + x_U),
            PerturbationLpNorm(x_L=x_L, x_U=x_U)
        )
        # cells = BoundedTensor(
        #     cells, ptb=PerturbationLpNorm(, x_l)
        # )

        # run training for n_epochs
        progress_bar = tqdm.tqdm(range(n_epochs))
        for epoch in progress_bar:
            # reset the gradients
            optimizer.zero_grad()

            # sample the batches
            with torch.no_grad():
                x_0 = initial_set.sample(batch_size)
                x_u = unsafe_set.sample(batch_size)
                x_inner = target_set.sample(batch_size)
                global_batch = global_set.sample(large_sample_size)
                global_values = v(global_batch)
                band_filter = torch.logical_and(
                    global_values <= beta_ra,
                    global_values > alpha_s
                ).squeeze(-1)
                x_decrease = global_batch[band_filter]
                if x_decrease.shape[0] > batch_size:
                    x_decrease = x_decrease.narrow(0, 0, batch_size)
                outer_filter = torch.logical_and(
                    (global_values <= beta_ra).squeeze(-1),
                    torch.logical_not(target_set.contains(global_batch))
                )
                x_outer = global_batch[outer_filter]
                # print(x_decrease.shape[0])

            # find the loss over the initial set points
            init_loss = torch.clamp(v(x_0) - alpha_ra, min=0.0).sum()

            # find the loss over the safety set points
            safety_loss = torch.clamp(beta_ra - v(x_u), min=0.0).sum()

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
            goal_loss_in = torch.clamp(v(x_inner) - beta_s, min=0.0).sum()

            # goal_loss_in = 0.0

            # find the loss outside of the target set
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
            # x_outer = non_target_area.contains(batch)

            # print(x_outer, V(x_outer))
            if torch.numel(x_outer) != 0:
                # print(x_outer)
                goal_loss_out = torch.clamp(beta_s - v(x_outer), min=0.0).sum()
            else:
                goal_loss_out = 0.0
            # goal_loss_out = 0.0

            # find the loss for the infinitesimal generator for reach-avoid
            if torch.numel(x_decrease) != 0:
                gen_values = generator(x_decrease)
                decrease_loss = torch.clamp(gen_values + zeta, min=0.0).sum()
                decrease_loss *= decrease_lambda
            else:
                decrease_loss = 0.0

            # find the loss regularizer
            regularizer = regularizer_lambda
            for layer in v.children():
                if isinstance(layer, torch.nn.Linear):
                    regularizer *= torch.linalg.matrix_norm(
                        layer.weight,
                        ord=torch.inf
                    )

            # find the total loss
            loss = init_loss + safety_loss + \
                goal_loss_out + goal_loss_in + \
                decrease_loss + \
                regularizer  # + \
            # batch_size * V(x_star)

            # update the progress bar with the loss information
            progress_bar.set_description(
                f"L0:{init_loss: 6.3f}, "
                f"Lu:{safety_loss: 8.3f}, "
                f"Lgo:{goal_loss_out: 6.3f}, "
                f"Lgi:{goal_loss_in: 6.3f}, "
                f"Ld:{decrease_loss: 8.3f}, "
                f"reg:{regularizer.item(): 6.3f}"
            )

            # if the loss is scalar (i.e., a batch sample was not
            # representative), go to the next step
            if isinstance(loss, float):
                continue

            # do the gradient step
            torch.nn.utils.clip_grad_norm_(v.parameters(), 1.0)
            loss.backward()
            optimizer.step()

            # do the gradient step
            if (epoch + 1) % verify_every_n == 0 or epoch + 1 == n_epochs:
                # verify
                print("=== Verification phase ===")

                # Verification step 1. Find sublevel set covers with perturbation analysis
                cell_lb, cell_ub = self.level_verifier.compute_bounds(
                    cells,
                    method="IBP"
                )

                # Verification step 2. reach-avoid probability
                # first, upper bound the certificate value at the initial states
                init_upper = torch.max(
                    cell_ub[initial_set.contains(cells), :]
                ).item()
                print(f"init: {init_upper}")

                # next, lower bound the certificate value at the unsafe states
                unsafe_lower = torch.min(
                    cell_lb[unsafe_set.contains(cells), :]
                ).item()
                print(f"unsafe: {unsafe_lower}")

                # compute the estimated reach-avoid probability
                # print(init_upper, unsafe_lower)
                if unsafe_lower <= 0.0:
                    unsafe_lower = 0.0
                    prob_ra_estimate = 0.0
                else:
                    prob_ra_estimate = max(1.0 - init_upper/unsafe_lower, 0.0)
                print(
                    f"Reach-avoid condition is satisfied with "
                    f"probability at least {prob_ra_estimate: 5.3f}."
                )

                # If it is higher than the specified probability, we can lower
                # the bounds. This helps with keeping the sub-sub_beta_ra_set.threshold set small
                # speeding up the computations. We changes the values slowly
                # so that the probability stays high; in particular, by changing
                # beta based on the previous alpha values, we make sure it is
                # only going down in a safer manner.
                # prob_threshold = (1.0 - prob_ra) * annealing
                # if 1.0 - prob_ra_estimate < prob_threshold:
                #     sub_beta_ra_set.threshold = min(
                #         sub_beta_ra_set.threshold,
                #         sub_alpha_ra_set.threshold / (1.0 - prob_ra)
                #     )
                #     sub_alpha_ra_set.threshold = lerp(
                #         sub_alpha_ra_set.threshold, max(
                #             init_upper,
                #             sub_beta_s_set.threshold
                #         ),
                #         0.5
                #     )
                # elif prob_ra_estimate < prob_ra:
                #     sub_beta_ra_set.threshold = max(
                #         sub_beta_ra_set.threshold,
                #         sub_alpha_ra_set.threshold / (1.0 - prob_ra)
                #     )

                # Verification step 3. stay probability
                # first, find a new alpha_s level if needed; we use the largest
                # upper bound of a cell, as all of the cell is then within this
                # sublevel set
                alpha_s_candidate = torch.min(
                    cell_ub[target_set.contains(cells), :]
                ).item()
                beta_s_candidate_in = torch.max(
                    cell_ub[target_set.contains(cells), :]
                ).item()
                # outer_cells = cell_lb[non_target_area.contains(cells), :]
                # if torch.numel(outer_cells) > 0:
                #     beta_s_candidate_out = torch.min(
                #         cell_lb[non_target_area.contains(cells), :]
                #     ).item()
                # else:
                #     beta_s_candidate_out = beta_s_candidate_in
                # target_upper = self._bound_estimate(
                #     [[0.0, 0.0]],
                #     [[-2.0, -torch.pi/3.0]],
                #     [[2.0, torch.pi/3.0]],
                #     bound_lower=False,
                #     method="alpha-CROWN"
                # )[1].item()
                # print(beta_s_candidate_in, beta_s_candidate_out, target_upper)
                beta_s_candidate = beta_s_candidate_in
                # sub_alpha_s_set.threshold = alpha_s_candidate
                # sub_beta_s_set.threshold = beta_s_candidate
                # min(beta_s_candidate_in, beta_s_candidate_out)
                print(
                    f"alpha_s: {alpha_s_candidate}, beta_s: {beta_s_candidate}")
                # if beta_s_candidate < sub_alpha_ra_set.threshold:
                #     sub_beta_s_set.threshold = lerp(
                #         beta_s_candidate,
                #         sub_beta_s_set.threshold,
                #         0.5
                #     )
                #     sub_alpha_s_set.threshold = sub_beta_s_set.threshold * (
                #         1.0 - prob_stay)

                prob_s_estimate = max(
                    1.0 - alpha_s_candidate/beta_s_candidate, 0.0)
                print(
                    f"Stay condition is satisfied with "
                    f"probability at least {prob_s_estimate: 5.3f}."
                )
                # if prob_s_estimate < self.specification.stay_probability:
                #     if beta_s_candidate < sub_alpha_ra_set.threshold:
                #         sub_beta_s_set.threshold = beta_s_candidate
                #     sub_alpha_s_set.threshold = sub_beta_s_set.threshold * \
                #         (1.0 - prob_s_estimate)

                mask = torch.logical_and(
                    target_set.contains(cells),
                    (cell_ub >= beta_s).squeeze()
                )
                x_inner_counterexamples = cells[mask, :]
                n_counterexamples = x_inner_counterexamples.shape[0]
                if n_counterexamples > 0:
                    print(
                        f"Found {n_counterexamples} potential "
                        "goal condition violations inside of the target set."
                    )

                # mask = torch.logical_and(
                #     spec.target_set.complement.contains(cells),
                #     (cell_lb <= sub_beta_s_set.threshold).squeeze()
                # )
                # x_outer_counterexamples = cells[mask, :]
                # n_counterexamples = x_outer_counterexamples.shape[0]
                # if n_counterexamples > 0:
                #     print(
                #         f"Found {n_counterexamples} potential "
                #         "goal condition violations outside of the target set."
                #     )

                # Verification step 4. decrease condition
                mask = torch.logical_and(
                    cell_lb > alpha_s,
                    cell_ub <= beta_ra
                ).squeeze()
                decrease_cells = cells[mask, :]
                # print(sub_alpha_s_set.threshold, sub_beta_ra_set.threshold)
                if torch.numel(decrease_cells) > 0:
                    bounded_decrease_cells = BoundedTensor(
                        decrease_cells,
                        PerturbationLpNorm(
                            x_L=cells.ptb.x_L[mask, :],
                            x_U=cells.ptb.x_U[mask, :]
                        )
                    )
                    _, ub = self.level_verifier.compute_bounds(
                        bounded_decrease_cells,
                        bound_lower=False,
                        method="IBP"
                    )
                    mask = ub.squeeze() >= 0.0
                    # print(decrease_cells[mask, :])
                    decrease_counterexamples = decrease_cells[mask, :]
                    n_decrease_counterexamples = decrease_counterexamples.shape[0]
                    if n_decrease_counterexamples > 0:
                        # zeta += lerp(zeta, torch.max(ub).item(), 0.1)
                        # else:
                        print(
                            f"Found {n_decrease_counterexamples} "
                            "potential decrease condition violations. "
                            f"Maximum value is {torch.max(ub).item(): 5.3f}. "
                            f"zeta is now {zeta: 5.3f}"
                        )

                # sub_beta_s_set.threshold = max(
                #     beta_s_candidate_in, beta_s_candidate_out)
                # if sub_alpha_ra_set.threshold < sub_beta_s_set.threshold:
                #     sub_alpha_ra_set.threshold = sub_beta_s_set.threshold * 2.0
                # if sub_beta_ra_set.threshold < sub_alpha_ra_set.threshold / (1 - self.specification.reach_avoid_probability):
                #     sub_beta_ra_set.threshold = sub_alpha_ra_set.threshold / \
                #         (1 - self.specification.reach_avoid_probability)
                #     sub_alpha_s_set.threshold = sub_beta_s_set.threshold * \
                #         (1 - self.specification.stay_probability)

                print(
                    f"beta_ra is {beta_ra: 5.3f}, "
                    f"alpha_ra is {alpha_ra: 5.3f}, "
                    f"beta_s is {beta_s: 5.3f}, "
                    f"alpha_s is {alpha_s: 5.3f}."
                )

                if n_counterexamples == 0 and n_decrease_counterexamples == 0 and \
                        prob_ra_estimate > self.specification.reach_avoid_probability and \
                        prob_s_estimate > self.specification.stay_probability:
                    return True, prob_ra_estimate, prob_s_estimate

        return False  # , prob_ra_estimate, prob_s_estimate
