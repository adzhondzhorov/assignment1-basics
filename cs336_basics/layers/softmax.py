import torch


def softmax(t: torch.Tensor, dim: int) -> torch.Tensor:
    t_max = t.amax(dim).unsqueeze(dim)
    t_norm = t - t_max
    t_norm_exp = t_norm.exp()
    t_norm_exp_sum = t_norm_exp.sum(dim).unsqueeze(dim)
    t_softmax = t_norm_exp / t_norm_exp_sum
    return t_softmax


def softmax_with_temp(t: torch.Tensor, dim: int, temperature: float) -> torch.Tensor:
    t_max = t.amax(dim).unsqueeze(dim)
    t_norm = t - t_max
    t_norm_exp = (t_norm / temperature).exp()
    t_norm_exp_sum = t_norm_exp.sum(dim).unsqueeze(dim)
    t_softmax = t_norm_exp / t_norm_exp_sum
    return t_softmax
