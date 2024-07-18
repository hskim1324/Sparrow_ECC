import math

import torch
from torch import Tensor

# This encoder is for Hamming(7, 4) encoding.
def hamming_encoder(n: int, k: int, input: Tensor) -> Tensor:
    
    if (n - k) != int(math.ceil(math.log(n, 2))):
        raise Exception('Not SEC encodable')

    # We assume a float32 Tensor input.
    data = input.view(torch.int32)
    data = data.view(data.numel())

    bits = []
    # Max exponent
    # We assume data is stored in MSBs starting from [28].
    for i in range(k):
        bits.append(((data&(1 << (28 - i))) >> (28 - i))&1)

    # Max exponent
    p1 = bits[0]^bits[1]^bits[3]
    p2 = bits[0]^bits[2]^bits[3]
    p3 = bits[1]^bits[2]^bits[3]

    data[True] = data[True] | (p1[True] << 23)
    data[True] = data[True] | (p2[True] << 22)
    data[True] = data[True] | (p3[True] << 21)

    del p1
    del p2
    del p3

    torch.cuda.empty_cache()

    return input

# This decoder is for Hamming(7, 4) decoding.
def hamming_decoder(n: int, k: int, input: Tensor) -> Tensor:
    
    if (n - k) != int(math.ceil(math.log(n, 2))):
        raise Exception('Not SEC encodable')
    
    # We assume a float32 Tensor input.
    data = input.view(torch.int32)
    data = data.view(data.numel())

    bits = []

    # Max exponent
    # [_|_|_|d]_[d|d|d|_]_[p1|p2|p3|_]
    # We assume data is in MSBs starting from [28].
    for i in range(k):
        bits.append(((data&(1 << (28 - i))) >> (28 - i))&1)
        bits[i] = bits[i].to(device=torch.device('cuda:0'), dtype=torch.int32)
    
    # parity
    bits.append(((data&(1 << 23)) >> 23)&1)
    bits.append(((data&(1 << 22)) >> 22)&1)
    bits.append(((data&(1 << 21)) >> 21)&1)

    s1 = bits[0]^bits[1]^bits[3]^bits[4]
    s2 = (bits[0]^bits[2]^bits[3]^bits[5]) << 1
    s3 = (bits[1]^bits[2]^bits[3]^bits[6]) << 2

    syndrome = s1 | s2 | s3

    # SEC
    data[syndrome==0b111] = (data[syndrome==0b111]^0x02000000) & 0xFF1FE000
    data[syndrome==0b110] = (data[syndrome==0b110]^0x04000000) & 0xFF1FE000
    data[syndrome==0b101] = (data[syndrome==0b101]^0x08000000) & 0xFF1FE000
    data[syndrome==0b100] = data[syndrome==0b100] & 0xFF1FE000 # p3
    data[syndrome==0b011] = (data[syndrome==0b011]^0x10000000) & 0xFF1FE000
    data[syndrome==0b010] = data[syndrome==0b010] & 0xFF1FE000 # p2
    data[syndrome==0b001] = data[syndrome==0b001] & 0xFF1FE000 # p1

    # No error
    data[syndrome==0b000] = data[syndrome==0b000] & 0xFF1FE000

    del s1
    del s2
    del s3
    del syndrome

    torch.cuda.empty_cache()

    return input