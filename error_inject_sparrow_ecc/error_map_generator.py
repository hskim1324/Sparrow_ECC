import numpy as np

import torch
from torch import Tensor

def generate(BER: float, block_count: int, block_size: int):
    
    error_count = 0

    # Whole error map (In Max exponent scheme, parity and data are unseparable.)
    error_map = torch.zeros(block_count * block_size, dtype=torch.int32)
    error_map = error_map.to(device=torch.device('cuda:0'), dtype=torch.int32)

    if BER==0:
        return error_map, 0
    
    weight = int(np.reciprocal(BER))

    bit_count = 18
    if block_size == 16:
        bit_count = 19

    for i in range(bit_count):
        error_map_temp = torch.randint(1, weight+1, (1, block_count * block_size))
        error_map_temp = error_map_temp.to(device=torch.device('cuda:0'), dtype=torch.int32)
        error_map_temp = error_map_temp.flatten()

        # Error = 1
        error_map_temp[error_map_temp==1] = 1
        error_map_temp[error_map_temp!=1] = 0

        error_count += torch.count_nonzero(error_map_temp)

        error_map_temp[True] = error_map_temp[True] << i
        error_map[True] = error_map[True] + error_map_temp[True]

        del error_map_temp
    
    error_map[True] = error_map[True] << (32 - bit_count)
    
    return error_map, error_count