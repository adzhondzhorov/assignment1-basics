import numpy as np
import torch

from cs336_basics.layers.parameter_initialization import init_weight


class SwiGLUPositionWiseFFN(torch.nn.Module):
    EPS = 1e-5

    def __init__(
        self,
        features: int,
        d_ffn: int = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        d_ffn = d_ffn or int(8 / 3 * features) // 16 * 16

        self.w1 = init_weight(d_ffn, features, device, dtype)
        self.w3 = init_weight(d_ffn, features, device, dtype)
        self.w2 = init_weight(features, d_ffn, device, dtype)

    @classmethod
    def _silu(self, a: torch.Tensor):
        return a * torch.sigmoid(a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        return (self._silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T


class SiLUPositionWiseFFN(torch.nn.Module):
    EPS = 1e-5

    def __init__(
        self,
        features: int,
        d_ffn: int = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        d_ffn = d_ffn or int(8 / 3 * features) // 16 * 16

        self.w1 = init_weight(d_ffn, features, device, dtype)
        self.w2 = init_weight(features, d_ffn, device, dtype)

    @classmethod
    def _silu(self, a: torch.Tensor):
        return a * torch.sigmoid(a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        return self._silu(x @ self.w1.T) @ self.w2.T
