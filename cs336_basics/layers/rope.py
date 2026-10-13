import torch
import einops as eo


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self, theta: float, d_qk: int, max_seq_len: int, device: torch.device = None
    ):
        super().__init__()
        self.d_qk = d_qk
        rsd = {}
        rs = []
        for i in range(max_seq_len):
            rs_internal = []
            for k in range(1, d_qk // 2 + 1):
                power = (2 * k - 2) / d_qk
                angle = i / (theta**power)
                theta_ik = torch.Tensor([angle])
                cos_ik = torch.cos(theta_ik)
                sin_ik = torch.sin(theta_ik)
                r = [[cos_ik, -sin_ik], [sin_ik, cos_ik]]
                rsd[(i, k)] = torch.tensor(r)
                rs_internal.append(r)
            rs.append(rs_internal)
        self.rs = torch.tensor(rs, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_reshaped = eo.rearrange(
            x, "... (d_model_div2 two) -> ... d_model_div2 two 1", two=2
        )
        rs_selected = self.rs[token_positions, :, :, :]
        n = x_reshaped.ndim - rs_selected.ndim
        rs_selected = eo.rearrange(rs_selected, "... ->" + "1 " * n + " ...")

        x_selected = x_reshaped[..., token_positions, :, :, :]
        result = rs_selected @ x_selected

        return eo.rearrange(
            result, "... d_model_div_2 two 1 -> ... (d_model_div_2 two)", two=2
        )
