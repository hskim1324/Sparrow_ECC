Sparrow ECC
===========

Python implementation of Sparrow ECC (ISLPED 2024)

> Hoseok Kim, Seung Hun Choi, Young-Ho Gong, Joonho Kong, and Sung Woo Chung. 2024. Sparrow ECC: A Lightweight ECC Approach for HBM Refresh Reduction towards Energy-efficient DNN Inference. In *ISLPED*.

## Structure

    .
    ├── error_inject_sparrow_ecc
    │   ├── arg_parser.py               # Argument parser for inference example code
    │   ├── decoder.py                  # Sparrow ECC decoder implementation
    │   ├── encoder.py                  # Sparrow ECC encoder implementation
    │   ├── error_inject.py             # Inject errors to model weights
    │   ├── error_map_generator.py      # Generate error bit map for injection
    │   ├── hamming.py                  # Hamming(7,4) encoder/decoder implementation
    │   └── inference_example.py        # Error injected inference example code
    ├── exponent_difference_analysis
    │   └── analyze.py
    ├── example.py
    ├── LICENSE
    └── README.md

## Requirements
* Python>=3.8
* Pytorch>=1.13.0
* openpyxl>=3.1.2 (only used to record results, optional)
* NumPy

## Error Injected Inference Example
We provide the Pytorch implementation of Sparrow ECC as well as an inference example code to evaluate the accuracy of _ResNet50_ depending on bit error rate (BER). In this example, we 
* Encode _ResNet50_ pretrained weights provided by Pytorch using the _Sparrow ECC encoder_
* To the encoded weights, inject random bit errors corresponding to a given bit error rate (BER)
* Decode the weights to correct bit errors using the _Sparrow ECC decoder_
* Evaluate image classification accuracy on the ImageNet validation dataset.

First, change your directory to:

```bash
cd error_inject_sparrow_ecc
```

Then, fill in the path to your imagenet validation dataset in `inference_example.py`

For error injected inference evaluation, we support all combinations of `BERs (from 0 to 1)` and `1->0 bit error proportions (from 0% to 100%)`.

For example, to run inference with `BER = 1e-3` and `1->0 error proportion = 1%`, run

```bash
python inference_example.py --BER 1e-3 --zero_to_one 0.01
```

This code supports `FP32`, `FP16`, and `BFLOAT16` based models with some minor changes. (default: `FP32`)

For more details, please refer to the paper.

## Exponent Difference Analysis
We provide the Python code used to analyze the exponent difference pattern in DNN models. 

The code analyzes the distribution of `maximum exponent - minimum exponent` values within weight blocks (64 consecutive weights)

First, change your directory to:

```bash
cd exponent_difference_analysis
```

Then, fill in the path to the model weight file you wish to analyze in ```analyze.py```, and run

```bash
python analyze.py
```

Which produces an excel file containing the exponent difference count and proportion results. 

You may change the weight block size by simply changing the `block_size` value in `analyze.py` (e.g., 64->32->16->8...)

This code supports `FP32`, `FP16`, and `BFLOAT16` based models with some minor changes. (default: `FP32`)

For more details, please refer to the paper.