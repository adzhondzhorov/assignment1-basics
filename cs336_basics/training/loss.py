import torch
import einops as eo


def cross_entropy(logits: torch.Tensor, y: torch.Tensor):
    norm_logits = logits - eo.reduce(logits, "... dim -> ... 1", "max")
    logits_exp_sum = eo.reduce(norm_logits.exp(), "... dim -> ... 1", "sum")

    log_logits_exp_sum = torch.log(logits_exp_sum)

    logloss = (norm_logits - log_logits_exp_sum).gather(-1, y[..., None]).squeeze(-1)

    return -eo.reduce(logloss, "... -> 1", "mean")
