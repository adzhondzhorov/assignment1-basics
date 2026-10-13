import torch


class Embedding(torch.nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        t = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        self.embedding_matrix = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(t, 0, 1, -3, 3)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]
