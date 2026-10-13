from typing import Callable, Optional

import torch


class SGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.0001):
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for param_group in self.param_groups:
            lr = param_group["lr"]
            for p in param_group["params"]:
                if p.grad is None:
                    continue
                p.data -= lr * p.grad.data
        return loss


# weights = torch.nn.Parameter(10 * torch.randn((10, 10)))
# opt = SGDOptimizer([weights], 10)

# for _ in range(100):
#     opt.zero_grad()
#     loss = (weights**2).mean()
#     print(loss.cpu().item())
#     loss.backward()
#     opt.step()
