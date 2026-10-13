import math


def get_lr_cosine_schedule(
    t: int, alpha_max: float, alpha_min: float, t_warmup: int, t_final: int
):
    if t < t_warmup:
        return alpha_max * t / t_warmup
    elif t > t_final:
        return alpha_min

    return (
        alpha_min
        + (alpha_max - alpha_min)
        * (1 + math.cos(math.pi * (t - t_warmup) / (t_final - t_warmup)))
        / 2
    )
