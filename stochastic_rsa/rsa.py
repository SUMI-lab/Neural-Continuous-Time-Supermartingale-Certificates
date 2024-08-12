from typing import Sequence
import torch
import tqdm
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from controlled_sde import ControlledSDE
from .specification import Specification
from .nets import CertificateModule, GeneratorModule, CertificateModuleWithDerivatives
from .cells import AdaptiveCellSystem


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
              verification_slack: float = 0.5,
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
        certificate = self.net   # this is V(t, x) in the paper
        certificate.train(True)  # make sure it's in training mode
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        # compute the threshold constants
        alpha_ra = starting_level
        beta_ra = alpha_ra / (1.0 - prob_ra) / verification_slack
        beta_s = alpha_ra * 0.9
        alpha_s = beta_s * (1.0 - prob_s) * verification_slack
        zeta_counter = zeta

        # x_star = torch.zeros((1, n_dim), device=self.device)

        # initialize the counterexample collections
        decrease_counterexamples = torch.empty((0, n_dim), device=self.device)
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
        cell_magnitudes = 0.5 * global_set.magnitudes / verifier_mesh_size
        # cells = BoundedTensor(
        #     cells, ptb=PerturbationLpNorm(, x_l)
        # )

        # run training for n_epochs
        progress_bar = tqdm.tqdm(range(n_epochs))
        for epoch in progress_bar:
            # reset the gradients
            optimizer.zero_grad()

            # sample the batches
            # with torch.no_grad():
            # x_0 = initial_set.sample(batch_size)
            # x_u = unsafe_set.sample(batch_size)
            # x_inner = target_set.sample(batch_size)
            # global_batch = global_set.sample(large_sample_size)
            # global_values = v(global_batch)
            # band_filter = torch.logical_and(
            #     global_values <= beta_ra,
            #     global_values > alpha_s
            # ).squeeze(-1)
            # x_decrease = global_batch[band_filter]
            # if x_decrease.shape[0] > batch_size:
            #     x_decrease = x_decrease.narrow(0, 0, batch_size)
            # outer_filter = torch.logical_and(
            #     (global_values <= beta_ra).squeeze(-1),
            #     torch.logical_not(target_set.contains(global_batch))
            # )
            # x_outer = global_batch[outer_filter]
            # print(x_decrease.shape[0])

            # sample a batch and compute the values for it
            batch = global_set.sample(batch_size)
            v = certificate(batch)

            # filter indices
            x_0 = initial_set.contains(batch)
            x_u = unsafe_set.contains(batch)
            x_g = target_set.contains(batch)
            x_d = torch.logical_and(
                v <= beta_ra,
                v > alpha_s
            ).squeeze(-1)

            # find the loss over the initial set points
            init_loss = torch.clamp(v[x_0] - alpha_ra, min=0.0).sum()

            # find the loss over the safety set points
            safety_loss = torch.clamp(beta_ra - v[x_u], min=0.0).sum()

            # find the loss over the interior of the target set
            goal_loss = torch.clamp(v[x_g] - beta_s, min=0.0).sum()

            # find the loss for the infinitesimal generator for reach-avoid
            gen_values = generator(batch[x_d])
            decrease_loss = torch.clamp(gen_values + zeta, min=0.0).sum()
            # if torch.numel(decrease_counterexamples) > 0:
            #     # print(decrease_counterexamples)
            #     p = torch.ones((decrease_counterexamples.shape[0],))
            #     index = p.multinomial(num_samples=batch_size, replacement=True)
            #     decrease_loss += torch.clamp(
            #         generator(decrease_counterexamples[index]) + zeta_counter, min=0.0).sum()
            decrease_loss *= decrease_lambda

            # find the loss regularizer
            regularizer = regularizer_lambda
            for layer in certificate.children():
                if isinstance(layer, torch.nn.Linear):
                    regularizer *= torch.linalg.matrix_norm(
                        layer.weight,
                        ord=torch.inf
                    )

            # find the total loss
            loss = init_loss + safety_loss + goal_loss + decrease_loss
            loss += regularizer

            # update the progress bar with the loss information
            progress_bar.set_description(
                f"L0:{init_loss: 6.3f}, "
                f"Lu:{safety_loss: 8.3f}, "
                f"Lg:{goal_loss: 6.3f}, "
                f"Ld:{decrease_loss: 8.3f}, "
                f"reg:{regularizer.item(): 6.3f}"
            )

            # if the loss is scalar (i.e., a batch sample was not
            # representative), go to the next step
            if isinstance(loss, float):
                continue

            # do the gradient step
            torch.nn.utils.clip_grad_norm_(certificate.parameters(), 1.0)
            loss.backward()
            optimizer.step()

            # verify
            if (epoch + 1) % verify_every_n == 0 or epoch + 1 == n_epochs:
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

                # next, lower bound the certificate value at the unsafe states
                unsafe_lower = torch.min(
                    cell_lb[unsafe_set.contains(cells), :]
                ).item()

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

                # Verification step 3. stay probability
                # first, find a new alpha_s level if needed; we use the largest
                # upper bound of a cell, as all of the cell is then within this
                # sublevel set
                beta_s_candidate = torch.max(
                    cell_lb[target_set.boundary_contains(
                        cells,
                        cell_magnitudes
                    ), :]
                ).item()
                alpha_s_candidate = torch.min(
                    cell_ub[target_set.contains(cells), :]
                ).item()

                # print(
                #     f"alpha_s: {alpha_s_candidate}, beta_s: {beta_s_candidate}")

                prob_s_estimate = max(
                    1.0 - alpha_s_candidate/beta_s_candidate, 0.0)
                print(
                    f"Stay condition is satisfied with "
                    f"probability at least {prob_s_estimate: 5.3f}."
                )

                if 1.0 - prob_s_estimate > (1.0 - prob_s) * verification_slack:
                    beta_s = min(beta_s_candidate, alpha_ra * 0.9)
                    alpha_s = beta_s * (1.0 - prob_s) * verification_slack

                # Verification step 4. decrease condition
                mask = torch.logical_and(
                    cell_lb > alpha_s_candidate,
                    cell_ub <= beta_ra
                ).squeeze()
                decrease_cells = cells[mask, :]
                # print(sub_alpha_s_set.threshold, sub_beta_ra_set.threshold)
                if torch.numel(decrease_cells) > 0:

                    cell_system = AdaptiveCellSystem()
                    decrease_counterexamples = cell_system.verify(
                        self.decrease_verifier,
                        decrease_cells,
                        cell_magnitudes
                    )

                    n_decrease_counterexamples = decrease_counterexamples.shape[0]
                    if n_decrease_counterexamples > 0:

                        # zeta = lerp(zeta, torch.max(ub).item(), 0.1)
                        # else:
                        # zeta_counter = lerp(
                        # zeta_counter, torch.max(ub).item(), 0.1)
                        print(
                            f"Found {n_decrease_counterexamples} "
                            "potential decrease condition violations. "
                        )

                print(
                    f"beta_ra is {beta_ra: 5.3f}, "
                    f"alpha_ra is {alpha_ra: 5.3f}, "
                    f"beta_s is {beta_s: 5.3f}, "
                    f"alpha_s is {alpha_s: 5.3f}."
                )

                if n_decrease_counterexamples == 0 and \
                        prob_ra_estimate > self.specification.reach_avoid_probability and \
                        prob_s_estimate > self.specification.stay_probability:
                    return True, prob_ra_estimate, prob_s_estimate

        return False  # , prob_ra_estimate, prob_s_estimate
