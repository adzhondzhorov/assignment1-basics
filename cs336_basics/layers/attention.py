import math

import numpy as np
import torch
import einops as eo

from cs336_basics.layers.parameter_initialization import init_weight
from cs336_basics.layers.softmax import softmax
from cs336_basics.layers.rope import RotaryPositionalEmbedding


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    assert q.shape[-1] == k.shape[-1]
    d_qk = q.shape[-1]
    q_k = q @ eo.rearrange(k, "... d1 d2 -> ... d2 d1") / math.sqrt(d_qk)

    if mask is not None:
        # q_k[~mask] = -torch.inf
        q_k = q_k.masked_fill(~mask, -torch.inf)

    return softmax(q_k, -1) @ v


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_theta: float | None = None,
        rope_max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        d_qk = d_v = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads

        self.Wq = init_weight(d_qk * num_heads, d_model, device, dtype)
        self.Wk = init_weight(d_qk * num_heads, d_model, device, dtype)
        self.Wv = init_weight(d_v * num_heads, d_model, device, dtype)
        self.Wo = init_weight(d_model, d_v * num_heads, device, dtype)
        self.device = device
        self.rope = None
        if rope_theta is not None and rope_max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                rope_theta, d_qk, rope_max_seq_len, device
            )

    def forward(
        self,
        x: torch.Tensor,
        rope_token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # d_model = 64
        # num_heads = 4
        # d_qk = 16
        # x.shape = [4, 12, 64]
        # Wq.shape = Wk.shape = Wv.shape = [64, 64]
        q = x @ self.Wq.T
        k = x @ self.Wk.T
        v = x @ self.Wv.T
        # q/k/v.shape = [4, 12, 64]
        # seq_len = 12
        original = "batch seq_len (num_heads d_model_div_num_heads)"
        if self.rope is not None and rope_token_positions is not None:
            rope_token_positions = rope_token_positions.squeeze(0)
            rope_transformed = "batch num_heads seq_len d_model_div_num_heads"
            q = eo.rearrange(
                q, f"{original} -> {rope_transformed}", num_heads=self.num_heads
            )
            q = self.rope.forward(q, rope_token_positions)
            q = eo.rearrange(
                q, f"{rope_transformed} -> {original}", num_heads=self.num_heads
            )
            k = eo.rearrange(
                k, f"{original} -> {rope_transformed}", num_heads=self.num_heads
            )
            k = self.rope.forward(k, rope_token_positions)
            k = eo.rearrange(
                k, f"{rope_transformed} -> {original}", num_heads=self.num_heads
            )

        seq_len = x.shape[-2]
        batch_len = x.shape[0]

        mask = eo.repeat(
            torch.triu(torch.ones(seq_len, seq_len, dtype=bool, device=self.device)),
            "w d -> m n d w",
            m=self.num_heads,
            n=batch_len,
        )
        # mask = eo.repeat(torch.triu(torch.ones(seq_len, seq_len, dtype=bool, device=self.device)), "w d -> n d w", n=batch_len)

        pattern = f"{original} -> num_heads batch seq_len d_model_div_num_heads"
        qs = eo.rearrange(q, pattern, num_heads=self.num_heads)

        ks = eo.rearrange(k, pattern, num_heads=self.num_heads)
        vs = eo.rearrange(v, pattern, num_heads=self.num_heads)
        # mh = torch.cat(
        #     [attention(rqi, rki, vi, mask) for rqi, rki, vi in zip(qs, ks, vs)], dim=-1
        # )
        mh = attention(qs, ks, vs, mask)
        rev_pattern = f" num_heads batch seq_len d_model_div_num_heads -> {original}"
        mh = eo.rearrange(mh, rev_pattern)
        # raise KeyError

        return mh @ self.Wo.T
