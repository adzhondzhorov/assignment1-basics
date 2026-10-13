import torch

from cs336_basics.layers.softmax import softmax
from cs336_basics.layers.embedding import Embedding
from cs336_basics.layers.linear import Linear
from cs336_basics.layers.rmsnorm import RMSNorm
from cs336_basics.layers.transformer_block import TransformerBlock


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        num_layers: int,
        d_model: int,
        d_ffn: int,
        num_heads: int,
        rope_theta: float,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    d_ffn,
                    num_heads,
                    rope_theta,
                    context_len,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.rmsnorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x):
        x = self.embedding(x)
        for tb in self.transformer_blocks:
            x = tb(x)
        x = self.rmsnorm(x)
        x = self.linear(x)
        return x  #  softmax(x, -1)
