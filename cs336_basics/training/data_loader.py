import random

import torch
import numpy as np


def load_batches(data: np.array, batch_size: int, context_len: int, device: str):
    batches = [], []
    for _ in range(batch_size):
        start_idx = random.randint(0, len(data) - context_len - 1)
        for i in range(2):
            batches[i].append(data[start_idx + i : start_idx + context_len + i])
    return torch.tensor(batches).to(device)
