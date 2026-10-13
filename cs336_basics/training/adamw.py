from typing import Callable, Optional
import math

import torch

from cs336_basics.training.gradient_clipping import gradient_clipping
from cs336_basics.training.lr_scheduler import get_lr_cosine_schedule


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        alpha=1e-3,
        lambda_=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        lr_schedule=None,
    ):
        defaults = {
            "alpha": alpha,
            "lambda": lambda_,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "lr_schedule": lr_schedule,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for param_group in self.param_groups:
            alpha = param_group["alpha"]
            lambda_ = param_group["lambda"]
            beta1 = param_group["beta1"]
            beta2 = param_group["beta2"]
            epsilon = param_group["epsilon"]
            lr_schedule = param_group["lr_schedule"]

            params = gradient_clipping(param_group["params"])
            # params = param_group["params"]
            for p in params:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                g = p.grad.data
                if lr_schedule is not None and alpha is None:
                    alpha = get_lr_cosine_schedule(t, **lr_schedule)
                alpha_t = alpha * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= alpha * lambda_ * p.data

                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g**2
                p.data -= alpha_t * m / (torch.sqrt(v) + epsilon)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


# weights = torch.nn.Parameter(10 * torch.randn((10, 10)))
# opt = AdamW([weights], 10)

# for _ in range(100):
#     opt.zero_grad()
#     loss = (weights**2).mean()
#     print(loss.cpu().item())
#     loss.backward()
#     opt.step()


# m[t] = b * m[t-1] + (1-b)*g[t] = b**2 * m[t-2] + b*(1-b)*g[t-1] + (1-b)*g[t]
# = b**3 * m[t-3] + b*(1-b)*g[t-1] + (1-b)*g[t]


# m[t-1] = b * m[t-2] + (1-b)*g[t-1]
