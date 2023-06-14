import random

import numpy as np
import torch


class Average:
    def __init__(self, n=0, s=0) -> None:
        self.n = n
        self.s = s
    
    def update(self, v, n=1):
        self.n += n
        self.s += n * v
    
    def get_value(self):
        # if self.n == 0:
        #     assert self.s == 0
        #     return 0

        return self.s / self.n


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
