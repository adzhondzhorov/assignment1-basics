import torch
import einops as eo

eps = 1e-6


def gradient_clipping(params, max_l2_norm: float = 0.5):
    l2_norm_sum = 0
    for p in params:
        if p.grad is not None:
            l2_norm_sum += eo.reduce(p.grad**2, "... -> 1", "sum")
    l2_norm = torch.sqrt(l2_norm_sum)
    if l2_norm > max_l2_norm:
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * max_l2_norm / (l2_norm + eps)
    return params
