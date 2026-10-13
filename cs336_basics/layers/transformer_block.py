import torch

from cs336_basics.layers.attention import MultiHeadSelfAttention
from cs336_basics.layers.positionwise_feedforward import (
    SiLUPositionWiseFFN,
    SwiGLUPositionWiseFFN,
)
from cs336_basics.layers.rmsnorm import RMSNorm


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        num_heads: int,
        rope_theta: float,
        rope_max_sequence_len: float,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rmsnorm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadSelfAttention(
            d_model,
            num_heads,
            rope_theta,
            rope_max_sequence_len,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLUPositionWiseFFN(d_model, d_ffn, device=device, dtype=dtype)
        # self.ffn = SiLUPositionWiseFFN(d_model, d_ffn, device=device, dtype=dtype)

    def forward(self, x):
        seq_len = x.shape[1]
        x = x + self.mha(self.rmsnorm1(x), torch.arange(0, seq_len, device=x.device))
        x = x + self.ffn(self.rmsnorm2(x))
        return x
