import torch
from torch import Tensor

import hamming

def decode(all_params: Tensor, block_size: int):

    all_params = hamming.hamming_decoder(7, 4, all_params)

    rows = int(all_params.numel() / block_size)
    all_params = all_params.view(rows, block_size)

    # get max exponent bits
    max_expo_bits = torch.zeros((rows, 8), dtype=torch.int32, device=torch.device('cuda:0'))

    if block_size == 16:
        start_index = 0
        for i in range(2): # duplicate 2 times
            max_expo_bits += ((torch.narrow(all_params, -1, start_index, 8) & 0x00002000) >> 13)
            start_index += 8
        
        start_index = 0
        for i in range(1): # duplicate 1 time
            max_expo_bits += ((torch.narrow(all_params, -1, start_index, 8) & 0x00004000) >> 14)
    
    else:
        start_index = 0
        for i in range(int(block_size / 8) - 1): # duplicate 7 times
            max_expo_bits += ((torch.narrow(all_params, -1, start_index, 8) & 0x00004000) >> 14)
            start_index += 8
    
    max_expo_bits = max_expo_bits.float()

    if block_size == 64:
        max_expo_bits[True] = max_expo_bits[True] / 7
    else:
        max_expo_bits[True] = max_expo_bits[True] / 3
    
    max_expo_bits = max_expo_bits.round().int()

    max_expos = torch.zeros((rows, 1), dtype=torch.int32, device=torch.device('cuda:0'))
    for i in range(8):
        max_expos += (torch.narrow(max_expo_bits, -1, i, 1) << (7 - i))

    # get flag bits
    flag_bits_one = ((torch.narrow(all_params, -1, block_size - 8, 8) & 0x00004000) >> 14)
    flag_bits = torch.zeros((rows, 1), dtype=torch.int32, device=torch.device('cuda:0'))
    
    for i in range(7): # duplicate 7 times
        flag_bits += torch.narrow(flag_bits_one, -1, i, 1)
    
    flag_bits = flag_bits.float()
    flag_bits[True] = flag_bits[True] / 7
    flag_bits = flag_bits.round().int()

    # cat max exponents, flags block size times.
    cat_tuple = ()
    cat_tuple2 = ()
    for i in range(block_size):
        cat_tuple += (max_expos,)
        cat_tuple2 += (flag_bits,)
    
    max_expos = torch.cat(cat_tuple, -1)
    flag_bits = torch.cat(cat_tuple2, -1)

    # get sign bits
    sign_bits = ((all_params & 0x80000000) >> 31 & 1) + ((all_params & 0x40000000) >> 30 & 1) + ((all_params & 0x20000000) >> 29 & 1)
    sign_bits = sign_bits.float()
    sign_bits[True] = sign_bits[True] / 3
    sign_bits = sign_bits.round().int()
    sign_bits[True] = sign_bits[True] << 31

    # diff < 16 decoding
    # [s|e|e|e|_|e|e|e|e|_|e|m|m|m|_|m|m|m|m]
    # [s|s|s|e|_|e|e|e|m|_|0|0|0|m|_|m|m|m|m|_|m|F|F] F: max expo or flag
    all_params[flag_bits == 0] = (sign_bits[flag_bits == 0] & 0x80000000)\
                                | ((max_expos[flag_bits == 0] - ((all_params[flag_bits == 0] & 0x1E000000) >> 25)) << 23)\
                                | ((all_params[flag_bits == 0] & 0x01000000) >> 2)\
                                | ((all_params[flag_bits == 0] & 0x001F8000) << 1)

    # diff >= 16 decoding
    # [s|s|s|e|_|e|e|e|e|_|0|0|0|m|_|m|m|m|m|_|m|F|F] F: max expo or flag
    all_params[flag_bits != 0] = (sign_bits[flag_bits != 0] & 0x80000000)\
                                | ((max_expos[flag_bits != 0] - ((all_params[flag_bits != 0] & 0x1F000000) >> 24)) << 23)\
                                | ((all_params[flag_bits != 0] & 0x001F8000) << 2)

    all_params[True] = all_params[True] & 0xFFFF0000

    all_params = all_params.view(all_params.numel())

    del max_expo_bits
    del max_expos
    del flag_bits
    del sign_bits

    torch.cuda.empty_cache()
    
    return all_params