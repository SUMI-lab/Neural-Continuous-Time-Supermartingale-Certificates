import torch


class AABBSet():
    def __init__(self, boundaries, device="cpu"):
        super().__init__()
        if not isinstance(boundaries, torch.Tensor):
            boundaries = torch.tensor(
                boundaries,
                device=device,
                dtype=torch.float32
            )
        with torch.no_grad():
            ndim = boundaries.ndim
            assert ndim in (2, 3), \
                f"Boundaries should have 2 or 3 dimensions, got {ndim} instead."
            if ndim == 2:
                boundaries = boundaries.unsqueeze(0)
            self.boundaries = boundaries
            self.low = boundaries[:, 0, :]
            self.high = boundaries[:, 1, :]
            self.magnitudes = boundaries[:, 1, :] - boundaries[:, 0, :]
            self.weights = torch.prod(self.magnitudes, dim=1)
            self.n_sets = boundaries.shape[0]

    def sample(self, n: int):
        with torch.no_grad():
            sets_to_sample = torch.multinomial(
                self.weights, n, replacement=True
            )
            location = torch.index_select(
                self.boundaries[:, 0, :], 0, sets_to_sample
            ).unsqueeze(0)
            scale = torch.index_select(
                self.magnitudes, 0, sets_to_sample
            ).unsqueeze(0)
            rnd = torch.rand(scale.shape)
            return (location + rnd * scale).squeeze(0)

    def contains(self, x: torch.Tensor, margin=0.0):
        boundaries = self.boundaries.unsqueeze(0)
        x = x.unsqueeze(1).unsqueeze(-2)
        # the dimensions are 0: batch, 1: aabb, 2: high/low, 3: x coordinates
        return torch.any(
            torch.all(
                torch.logical_and(
                    x <= boundaries[:, :, 1, :] + margin,
                    x >= boundaries[:, :, 0, :] - margin
                ),
                dim=3
            ),
            dim=2
        ).squeeze(1)

    def boundary_contains(self, x: torch.Tensor, margin=0.0):
        boundaries = self.boundaries.unsqueeze(0)
        x = x.unsqueeze(1).unsqueeze(-2)
        margin = margin.unsqueeze(0).unsqueeze(-2)
        # the dimensions are 0: batch, 1: aabb, 2: high/low, 3: x coordinates
        return torch.any(
            torch.logical_and(
                torch.all(
                    torch.logical_and(
                        x <= boundaries[:, :, 1, :] + margin,
                        x >= boundaries[:, :, 0, :] - margin
                    ),
                    dim=3
                ),
                torch.any(
                    torch.logical_or(
                        x > boundaries[:, :, 1, :] - margin,
                        x < boundaries[:, :, 0, :] + margin
                    ),
                    dim=3
                )
            ),
            dim=2
        ).squeeze(1)
