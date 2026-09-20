import os
import random

import numpy as np
import torch


def set_seed(seed: int, set_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.accelerator.manual_seed(seed)
    torch.accelerator.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if set_deterministic:
        set_deterministic_algorithms()


def set_deterministic_algorithms():
    '''
    HCCL_DETERMINISTIC: is a deterministic switch in ops level, set it to 'True' to enable ops level deterministic,
        set it to 'False' to disable ops level deterministic.
    CLOSE_MATMUL_K_SHIFT: is a switch of matmul K-axis shift, set it to '1' to close matmul K-axis shift,
        set it to '0' to enable matmul K-axis shift.
    '''
    os.environ['HCCL_DETERMINISTIC'] = 'True'
    os.environ['CLOSE_MATMUL_K_SHIFT'] = '1'
    torch.use_deterministic_algorithms(True)
