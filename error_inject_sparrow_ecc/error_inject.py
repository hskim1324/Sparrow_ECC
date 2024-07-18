import os

import torch
from torch import Tensor

import error_map_generator
import encoder
import decoder

def inject(dict_params: dict, BER: str, block_size: int, model_name: str, zero_to_one: float=0):

    with torch.inference_mode():

        all_params_tuple = ()
        for name, _ in dict_params.items():
            if dict_params[name].dtype != torch.float32:
                dict_params[name] = dict_params[name]
                continue

            if 'weight' not in name:
                if 'bias' not in name:
                    dict_params[name] = dict_params[name]
                    continue

            temp = dict_params[name].view(torch.int32)
            temp = temp.view(dict_params[name].numel())

            all_params_tuple += (temp,)

            del temp
            torch.cuda.empty_cache()

        all_params = torch.cat(all_params_tuple)

        remainder = all_params.numel() % block_size
        remainder_zeros = torch.zeros(block_size-remainder, dtype=torch.int32, device=torch.device('cuda:0'))

        all_params = torch.cat((all_params, remainder_zeros))

        del remainder_zeros
        torch.cuda.empty_cache()

        all_params = all_params.view(torch.int32)

        block_count = int(all_params.numel() / block_size)

        all_params[True] = all_params[True] & 0xFFFF0000
        
        # Step 1: Generate error map for data, system ecc, on-die ecc
        # 1->0 error map and 0->1 error map are generated separately
        BER = float(BER)
        one_to_zero_BER = BER * (1 - zero_to_one)
        zero_to_one_BER = BER * zero_to_one
        
        one_to_zero_error_map, one_to_zero_error_count = error_map_generator.generate(one_to_zero_BER, block_count, block_size)
        zero_to_one_error_map, zero_to_one_error_count = error_map_generator.generate(zero_to_one_BER, block_count, block_size)

        error_count = one_to_zero_error_count + zero_to_one_error_count

        if error_count == 0:

            del all_params
            del one_to_zero_error_map
            del zero_to_one_error_map
            torch.cuda.empty_cache()

            return dict_params, False, BER

        else:
            if block_size == 16:
                BER = str(error_count / (all_params.numel() * 19))
            else:
                BER = str(error_count / (all_params.numel() * 18))

        cpt_file_name = 'all_params_dict_' + model_name + '_' + str(block_size) + '.pt'

        # Step 2: Encode
        if os.path.isfile(cpt_file_name):
            print('Load from checkpoint...')

            all_params_dict = torch.load(cpt_file_name)
            all_params_dict = all_params_dict['all_params_dict']
            all_params = all_params_dict['all_params']
        
        else:
            print('Start encoding...')

            all_params = encoder.encode(all_params, block_size)

            print('Encoding done...')

            all_params_dict = {'all_params': all_params}
            
            torch.save({
                'all_params_dict': all_params_dict,
            },
            cpt_file_name)

        # Step 3: Error inject
        # 1->0 error inject
        one_to_zero_error_map[True] = ~one_to_zero_error_map[True]
        all_params[True] = all_params[True] & one_to_zero_error_map[True]
        # 0->1 error inject
        all_params[True] = all_params[True] | zero_to_one_error_map[True]

        del one_to_zero_error_map
        del zero_to_one_error_map
        torch.cuda.empty_cache()

        print('Start decoding...')

        # Step 4: Decode
        all_params = decoder.decode(all_params, block_size)

        print('Decoding done...')

    start_index = 0
    for name, _ in dict_params.items():
        if dict_params[name].dtype != torch.float:
            dict_params[name] = dict_params[name]
            continue

        if 'weight' not in name:
            if 'bias' not in name:
                dict_params[name] = dict_params[name]
                continue

        temp = dict_params[name].view(torch.int32)
        temp = temp.view(dict_params[name].numel())

        temp2 = torch.narrow(all_params, -1, start_index, dict_params[name].numel())

        temp[True] = temp2[True]

        del temp2
        torch.cuda.empty_cache()

        start_index += dict_params[name].numel()

    # This function returns 
    # 1. error injected parameters
    # 2. bool that tells whether there are errors
    # 3. real 'calculated' bit error rate after injection
    return dict_params, True, BER