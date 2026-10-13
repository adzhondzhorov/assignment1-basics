import numpy as np
import torch


def init_weight(out_features, in_feataures, device, dtype):
    t = torch.empty((out_features, in_feataures), device=device, dtype=dtype)
    mean = 0
    std = np.sqrt(2 / (in_feataures + out_features))
    return torch.nn.Parameter(
        torch.nn.init.trunc_normal_(t, mean, std, -3 * std, 3 * std)
    )
