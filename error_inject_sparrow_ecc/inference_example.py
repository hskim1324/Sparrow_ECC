import time
import numpy as np

import torch, torchvision
from torch import Tensor

import arg_parser
import error_inject

parser = arg_parser.parser_init()
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# toggle tensorfloat32 in Ampere or later GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False

params = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
model = torchvision.models.resnet50(weights=params).to(device)
transform = params.transforms()

###########
# Fill in path to imagenet validation dataset
test_set = torchvision.datasets.ImageFolder('/PATH_TO_IMAGENET_VALIDATION_DATASET' , transform=transform)
###########
test_loader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=True, num_workers=16)

model.eval()

dict_params = model.state_dict()

start = time.time()

dict_params, is_error, BER = error_inject.inject(dict_params, args.BER, block_size=64, model_name='ResNet50', zero_to_one=float(args.zero_to_one))

if not is_error:
    print('There were no errors injected, use baseline accuracy')
    exit()

model.load_state_dict(dict_params, strict=False)
# Sparrow ECC is for Bfloat16 weights
model = model.to(torch.bfloat16)

# conduct inference accuracy evaluation
correct_top1 = 0
correct_top5 = 0
total = 0

with torch.inference_mode():
    for idx, (images, labels) in enumerate(test_loader):

        images = images.to(device)      # [10, 3, centercrop_size, centercrop_size]
        labels = labels.to(device)      # [10]

        # Bfloat16 inference
        images = images.to(torch.bfloat16)

        outputs = model(images)

        # ------------------------------------------------------------------------------
        # rank 1
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct_top1 += (pred == labels).sum().item()

        # ------------------------------------------------------------------------------
        # rank 5
        _, rank5 = outputs.topk(5, 1, True, True)
        rank5 = rank5.t()
        correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))

        # ------------------------------------------------------------------------------
        for k in range(6):
            correct_k = correct5[:k].contiguous().view(-1).float().sum(0, keepdim=True)

        correct_top5 += correct_k.item()

        # print accuracy per batch size(cumulative)
        print("step : {} / {}".format(idx + 1, len(test_set)/int(labels.size(0))))
        print("top-1 percentage :  {0:0.3f}%".format(correct_top1 / total * 100))
        print("top-5 percentage :  {0:0.3f}%".format(correct_top5 / total * 100))

end = time.time()

print("top-1 percentage :  {0:0.3f}%".format(correct_top1 / total * 100))
print("top-5 percentage :  {0:0.3f}%".format(correct_top5 / total * 100))
print("Elapsed time :  {0:0.3f}s".format(end - start))

torch.cuda.empty_cache()