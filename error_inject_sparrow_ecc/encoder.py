import torch
from torch import Tensor

import hamming

def encode(all_params: Tensor, block_size: int):

    rows = int(all_params.numel() / block_size)
    all_params = all_params.view(rows, block_size)

    exponents = (all_params & 0x7F800000) >> 23

    max_expos, _ = torch.max(exponents, -1, True)
    cat_tuple = ()
    for i in range(block_size):
        cat_tuple += (max_expos,)
    max_expos = torch.cat(cat_tuple, -1)

    exponents[exponents==0] = 0xFF

    min_expos, _ = torch.min(exponents, -1, True)
    cat_tuple = ()
    for i in range(block_size):
        cat_tuple += (min_expos,)
    min_expos = torch.cat(cat_tuple, -1)

    diff_expos = max_expos - min_expos
    diff_expos[diff_expos < 0] = 0

    all_params[diff_expos >= 16] = all_params[diff_expos >= 16] & 0xFFFE0000
    # Diff >= 32 values to smallest expressible value.
    all_params[(max_expos-((all_params&0x7F800000)>>23)) >= 32] = (all_params[(max_expos-((all_params&0x7F800000)>>23)) >= 32] & 0x80000000) | ((max_expos[(max_expos-((all_params&0x7F800000)>>23)) >= 32]-31) << 23)

    # conversion complete, start encoding
    
    # generate difference flags
    diff_expos[diff_expos < 16] = -1
    diff_expos[diff_expos >= 16] = -2

    # generate max exponent bits

    # 256-bit block encoding
    if block_size == 16:
        # first column (bit position 1)
        max_expo_bits_first = Tensor().int()
        max_expo_bits_first = max_expo_bits_first.to(device=torch.device('cuda:0'))

        for i in range(15, 7, -1):
            temp = torch.narrow(max_expos, -1, 0, 1)
            temp = (temp & (2**(i%8))) >> (i%8)
            if max_expo_bits_first.size() == 0:
                max_expo_bits_first = temp.clone().detach()
            else:
                max_expo_bits_first = torch.cat((max_expo_bits_first, temp), -1)
        
        temp = torch.narrow(diff_expos, -1, 0, 1).clone().detach()
        temp[temp == -1] = 0
        temp[temp == -2] = 1
        for i in range(8):
            max_expo_bits_first = torch.cat((max_expo_bits_first, temp), -1)
        
        # second column (bit position 0)
        max_expo_bits_second = Tensor().int()
        max_expo_bits_second = max_expo_bits_second.to(device=torch.device('cuda:0'))
        
        for i in range(15, -1, -1):
            temp = torch.narrow(max_expos, -1, 0, 1)
            temp = (temp & (2**(i%8))) >> (i%8)
            if max_expo_bits_second.size() == 0:
                max_expo_bits_second = temp.clone().detach()
            else:
                max_expo_bits_second = torch.cat((max_expo_bits_second, temp), -1)

        # diff < 16 encoding
        # [s|e|e|e|_|e|e|e|e|_|e|m|m|m|_|m|m|m|m]
        # [s|s|s|e|_|e|e|e|m|_|0|0|0|m|_|m|m|m|m|_|m|F|F] F: max expo or flag
        all_params[diff_expos == -1] = (all_params[diff_expos == -1] & 0x80000000)\
                                    | ((all_params[diff_expos == -1] & 0x80000000) >> 1 & 0x40000000)\
                                    | ((all_params[diff_expos == -1] & 0x80000000) >> 2 & 0x20000000)\
                                    | ((max_expos[diff_expos == -1] - ((all_params[diff_expos == -1] & 0x7F800000) >> 23)) << 25)\
                                    | ((all_params[diff_expos == -1] & 0x00400000) << 2)\
                                    | ((all_params[diff_expos == -1] & 0x003F0000) >> 1)\
                                    | (max_expo_bits_first[diff_expos == -1] << 14)\
                                    | (max_expo_bits_second[diff_expos == -1] << 13)

        # diff >= 16 encoding
        # [s|s|s|e|_|e|e|e|e|_|0|0|0|m|_|m|m|m|m|_|m|F|F] F: max expo or flag
        all_params[diff_expos == -2] = (all_params[diff_expos == -2] & 0x80000000)\
                                    | ((all_params[diff_expos == -2] & 0x80000000) >> 1 & 0x40000000)\
                                    | ((all_params[diff_expos == -2] & 0x80000000) >> 2 & 0x20000000)\
                                    | ((max_expos[diff_expos == -2] - ((all_params[diff_expos == -2] & 0x7F800000) >> 23)) << 24)\
                                    | ((all_params[diff_expos == -2] & 0x007E0000) >> 2)\
                                    | (max_expo_bits_first[diff_expos == -2] << 14)\
                                    | (max_expo_bits_second[diff_expos == -2] << 13)
        
        del max_expo_bits_first
        del max_expo_bits_second
        del temp
    
    # 512-bit, 1024-bit block encoding
    else:
        max_expo_bits = Tensor().int()
        max_expo_bits = max_expo_bits.to(device=torch.device('cuda:0'))
        for i in range((block_size - 1), 7, -1):
            temp = torch.narrow(max_expos, -1, 0, 1)
            temp = (temp & (2**(i%8))) >> (i%8)
            if max_expo_bits.size() == 0:
                max_expo_bits = temp.clone().detach()
            else:
                max_expo_bits = torch.cat((max_expo_bits, temp), -1)
        
        temp = torch.narrow(diff_expos, -1, 0, 1).clone().detach()
        temp[temp == -1] = 0
        temp[temp == -2] = 1
        for i in range(8):
            max_expo_bits = torch.cat((max_expo_bits, temp), -1)
    
        # diff < 16 encoding
        # [s|e|e|e|_|e|e|e|e|_|e|m|m|m|_|m|m|m|m]
        # [s|s|s|e|_|e|e|e|m|_|0|0|0|m|_|m|m|m|m|_|m|F] F: max expo or flag
        all_params[diff_expos == -1] = (all_params[diff_expos == -1] & 0x80000000)\
                                    | ((all_params[diff_expos == -1] & 0x80000000) >> 1 & 0x40000000)\
                                    | ((all_params[diff_expos == -1] & 0x80000000) >> 2 & 0x20000000)\
                                    | ((max_expos[diff_expos == -1] - ((all_params[diff_expos == -1] & 0x7F800000) >> 23)) << 25)\
                                    | ((all_params[diff_expos == -1] & 0x00400000) << 2)\
                                    | ((all_params[diff_expos == -1] & 0x003F0000) >> 1)\
                                    | (max_expo_bits[diff_expos == -1] << 14)

        # diff >= 16 encoding
        # [s|s|s|e|_|e|e|e|e|_|0|0|0|m|_|m|m|m|m|_|m|F] F: max expo or flag
        all_params[diff_expos == -2] = (all_params[diff_expos == -2] & 0x80000000)\
                                    | ((all_params[diff_expos == -2] & 0x80000000) >> 1 & 0x40000000)\
                                    | ((all_params[diff_expos == -2] & 0x80000000) >> 2 & 0x20000000)\
                                    | ((max_expos[diff_expos == -2] - ((all_params[diff_expos == -2] & 0x7F800000) >> 23)) << 24)\
                                    | ((all_params[diff_expos == -2] & 0x007E0000) >> 2)\
                                    | (max_expo_bits[diff_expos == -2] << 14)
        
        del max_expo_bits
        del temp
    
    all_params = all_params.view(all_params.numel())
    all_params = hamming.hamming_encoder(7, 4, all_params)

    del exponents
    del max_expos
    del min_expos
    del diff_expos

    torch.cuda.empty_cache()

    return all_params