import torch
from auto_LiRPA import BoundedTensor, PerturbationLpNorm, BoundedModule


class AdaptiveCellSystem():
    def __init__(self, max_depth=10):
        super().__init__()
        self.max_depth = max_depth
        self.corners = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])

    def verify(
        self,
            verifier: BoundedModule,
            locations: torch.Tensor,
            magnitude: torch.Tensor,
            depth: int = 0
    ):
        bounded_cells = BoundedTensor(
            locations,
            PerturbationLpNorm(
                x_L=locations - magnitude,
                x_U=locations + magnitude,
            )
        )
        _, ub = verifier.compute_bounds(
            bounded_cells,
            bound_lower=False,
            method="IBP"
        )
        mask = ub.squeeze() >= 0.0
        counterexamples = locations[mask]
        print(
            f"Could not verify decrease at {counterexamples.shape[0]} cells. Splitting further")
        if depth < self.max_depth and torch.numel(counterexamples) > 0:
            new_cells = torch.empty((0, locations.shape[1]))
            half_m = 0.5 * magnitude
            for _, loc in enumerate(counterexamples):
                new_cells = torch.cat(
                    (new_cells, loc + half_m * self.corners),
                    dim=0
                )
            counterexamples = self.verify(
                verifier, new_cells, half_m, depth+1
            )
        return counterexamples
