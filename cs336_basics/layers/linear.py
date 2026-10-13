import numpy as np
import torch

from cs336_basics.layers.parameter_initialization import init_weight


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.W = init_weight(out_features, in_features, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T
