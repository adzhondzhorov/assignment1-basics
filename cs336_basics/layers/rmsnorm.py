import torch
import einops as eo


class RMSNorm(torch.nn.Module):
    EPS = 1e-5

    def __init__(
        self, in_features: int, device: torch.device = None, dtype: torch.dtype = None
    ):
        super().__init__()
        t = torch.ones((in_features,), device=device, dtype=dtype)
        self.g = torch.nn.Parameter(t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        denom = torch.sqrt(
            eo.reduce(x**2, "... d -> ... 1", "sum") / x.shape[-1] + self.EPS
        )
        x = x.to(in_dtype)
        return x * self.g / denom
