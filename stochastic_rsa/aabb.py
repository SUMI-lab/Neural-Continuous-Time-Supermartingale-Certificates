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
            self.low = boundaries[:, 0, :]
            self.high = boundaries[:, 1, :]
            self.magnitudes = self.high - self.low
            self.weights = torch.prod(self.magnitudes, dim=1)

    def sample(self, n: int):
        sets_to_sample = torch.multinomial(self.weights, n, replacement=True)
        location = torch.index_select(self.low, 0, sets_to_sample).unsqueeze(0)
        scale = torch.index_select(
            self.magnitudes, 0, sets_to_sample
        ).unsqueeze(0)
        rnd = torch.rand(scale.shape)
        return (location + rnd * scale).squeeze(0)

    def contains(self, x: torch.Tensor):
        x_tiled = x.unsqueeze(0).tile((self.high.shape[0], 1, 1))
        # print(x_tiled, self.high.unsqueeze(0), self.low.unsqueeze(0))
        # print(torch.logical_and(x_tiled <= self.high.unsqueeze(
        #     0), x_tiled >= self.low.unsqueeze(0)))
        return torch.any(torch.all(torch.logical_and(x_tiled <= self.high.unsqueeze(0), x_tiled >= self.low.unsqueeze(0)), dim=2), dim=0)
