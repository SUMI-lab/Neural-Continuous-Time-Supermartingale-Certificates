from typing import Sequence
import torch
import tqdm
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from controlled_sde import ControlledSDE
from .specification import Specification
from .nets import CertificateModule, GeneratorModule, CertificateModuleWithDerivatives
from .cells import CellVerificationSystem


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
              n_epochs: int = 1_000_000,
              batch_size: int = 256,
              lr: float = 1e-3,
              zeta: float = 1.0,
              verify_every_n: int = 1000,
              verification_slack: float = 2.0,
              regularizer_lambda: float = 1e-3,
              verifier_mesh_size: int = 400,
              max_depth: int = 4
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

        # initialize the levels (step 1 in the paper)
        alpha_ra = 1.0
        beta_ra = alpha_ra / (1.0 - prob_ra) * verification_slack
        beta_s = alpha_ra * 0.9
        alpha_s = beta_s * (1.0 - prob_s) / verification_slack

        # create the cells (step 2 in the paper)
        cells = torch.meshgrid(
            torch.linspace(global_set.low[0, 0],
                           global_set.high[0, 0], verifier_mesh_size + 1),
            torch.linspace(global_set.low[0, 1],
                           global_set.high[0, 1], verifier_mesh_size + 1),
            indexing='xy'
        )
        x_L = torch.cat(
            (torch.reshape(cells[0][:-1, :-1], (-1,)).unsqueeze(1),
             torch.reshape(cells[1][:-1, :-1], (-1,)).unsqueeze(1)),
            dim=1
        )  # lower cell bounds
        x_U = torch.cat(
            (torch.reshape(cells[0][1:, 1:], (-1,)).unsqueeze(1),
             torch.reshape(cells[1][1:, 1:], (-1,)).unsqueeze(1)),
            dim=1
        )  # upper cell bounds
        cells = BoundedTensor(
            0.5 * (x_L + x_U),
            PerturbationLpNorm(x_L=x_L, x_U=x_U)
        )  # create cells as bounded tensors for auto_LiRPA
        cell_magnitudes = 0.5 * global_set.magnitudes / verifier_mesh_size

        # run training for n_epochs
        progress_bar = tqdm.tqdm(range(n_epochs))
        for epoch in progress_bar:  # step 3 in the paper
            # reset the gradients
            optimizer.zero_grad()

            # sample a batch and compute the values for it (step 4 in the paper)
            batch = global_set.sample(batch_size)
            v = certificate(batch)

            # filter indices for different sets within the state space
            x_u = unsafe_set.contains(batch)
            x_0 = initial_set.contains(batch)
            x_g = target_set.contains(batch)
            x_d = torch.logical_and(v <= beta_ra, v > alpha_s).squeeze(-1)

            # find the loss over the safety set points - eq. (11)
            safety_loss = torch.clamp(beta_ra - v[x_u], min=0.0).sum()

            # find the loss over the initial set points - eq. (12)
            init_loss = torch.clamp(v[x_0] - alpha_ra, min=0.0).sum()

            # find the loss over the interior of the target set - eq. (13)
            goal_loss = torch.clamp(v[x_g] - beta_s, min=0.0).sum()

            # find the loss for the infinitesimal generator - eq. (14)
            gen_values = generator(batch[x_d])
            decrease_loss = torch.clamp(gen_values + zeta, min=0.0).sum()

            # find the loss regularizer - eq. (15)
            regularizer = regularizer_lambda
            for layer in certificate.children():
                if isinstance(layer, torch.nn.Linear):
                    regularizer *= torch.linalg.matrix_norm(
                        layer.weight,
                        ord=torch.inf
                    )

            # find the total loss - eq. (10) and step 5 of the algorithm
            loss = init_loss + safety_loss + goal_loss + decrease_loss
            loss += regularizer

            # update the progress bar with the loss information
            progress_bar.set_description(
                f"L0:{init_loss: 8.3f}, "
                f"Lu:{safety_loss: 8.3f}, "
                f"Lg:{goal_loss: 8.3f}, "
                f"Ld:{decrease_loss: 8.3f}, "
                f"reg:{regularizer.item(): 6.3f}"
            )

            # if somehow the loss is scalar (i.e., a batch sample was not
            # representative), go to the next step
            if isinstance(loss, float):
                continue

            # do the gradient step (step 6 of the algorithm)
            torch.nn.utils.clip_grad_norm_(certificate.parameters(), 1.0)
            loss.backward()
            optimizer.step()

            # verify (step 7 of the algorithm)
            if (epoch + 1) % verify_every_n == 0 or epoch + 1 == n_epochs:
                print("=== Verification phase ===")

                # Verification step 1. Find sublevel set covers
                # with interval bound propagation. This is step 8 of the
                # algorithm in the paper.
                cell_lb, cell_ub = self.level_verifier.compute_bounds(
                    cells,
                    method="IBP"
                )

                # Verification step 2. Reach-avoid probability. This is step
                # 9 of the algorithm.

                # first, upper bound the certificate value at the initial states
                init_upper = torch.max(
                    cell_ub[initial_set.contains(cells), :]
                ).item()

                # next, lower bound the certificate value at the unsafe states
                unsafe_cells = cell_lb[unsafe_set.contains(cells), :]
                if torch.numel(unsafe_cells) > 0:
                    unsafe_lower = torch.min(
                        cell_lb[unsafe_set.contains(cells), :]
                    ).item()
                else:
                    unsafe_lower = init_upper

                # compute the estimated reach-avoid probability, eq. (16)
                if unsafe_lower <= 0.0:
                    prob_ra_estimate = 0.0
                else:
                    prob_ra_estimate = max(1.0 - init_upper/unsafe_lower, 0.0)
                print(
                    f"Reach-avoid condition is satisfied with "
                    f"probability at least {prob_ra_estimate: 5.3f}."
                )
                if prob_ra_estimate < self.specification.reach_avoid_probability:
                    continue

                # Verification step 3. Stay probability. This is step 10.

                # first, find a new alpha_s level if needed; we use the largest
                # upper bound of a cell, as all of the cell is then within this
                # sublevel set
                alpha_s_candidate = torch.min(
                    cell_ub[target_set.contains(cells), :]
                ).item()
                beta_s_candidate = torch.max(
                    cell_lb[target_set.boundary_contains(
                        cells,
                        cell_magnitudes
                    ), :]
                ).item()

                # compute the estimated reach-avoid probability, eq. (17)
                prob_s_estimate = max(
                    1.0 - alpha_s_candidate/beta_s_candidate, 0.0)
                print(
                    f"Stay condition is satisfied with "
                    f"probability at least {prob_s_estimate: 5.3f}."
                )
                if prob_s_estimate < self.specification.stay_probability:
                    continue

                # Verification step 4. decrease condition.

                # find the decrease cells (18)
                mask = torch.logical_and(
                    cell_lb > alpha_s_candidate,
                    cell_ub <= beta_ra
                ).squeeze()
                decrease_cells = cells[mask, :]

                if torch.numel(decrease_cells) > 0:

                    # These are steps 11-18 put into a separate function
                    cell_system = CellVerificationSystem(max_depth)
                    decrease_counterexamples = cell_system.verify(
                        self.decrease_verifier,
                        decrease_cells,
                        cell_magnitudes
                    )
                    n_decrease_counterexamples = decrease_counterexamples.shape[0]
                    if n_decrease_counterexamples > 0:
                        print(
                            f"Found {n_decrease_counterexamples} "
                            "potential decrease condition violations. "
                        )
                else:
                    n_decrease_counterexamples = 0

                # Steps 19-20 of the algorithm
                if n_decrease_counterexamples == 0:
                    return True, epoch

        # Step 21 of the algorithm
        return False, n_epochs
