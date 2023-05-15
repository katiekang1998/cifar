import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
from custom_datasets import CIFAR10C
from trainer import AverageMeter, model_names
import numpy as np
import pickle


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

parser.add_argument('--run-name', dest='run_name',
                    type=str, default="")
parser.add_argument('--corruption-type', dest='corruption_type',
                    help='Type of corruption to add to evaluation images',
                    type=str, default="impulse_noise")

args = parser.parse_args()
run_name = args.run_name
corruption_type = args.corruption_type

checkpoint_name = "data/"+run_name+"/checkpoint.th"


model = torch.nn.DataParallel(resnet.__dict__['resnet20'](11))
model.cuda()


print("=> loading checkpoint '{}'".format(checkpoint_name))
checkpoint = torch.load(checkpoint_name)
start_epoch = checkpoint['epoch']
best_prec1 = checkpoint['best_prec1']
model.load_state_dict(checkpoint['state_dict'])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=128, shuffle=False,
    num_workers=4, pin_memory=True)



valc_loaders = []
for corruption_level in range(5):
    valc_loader = torch.utils.data.DataLoader(
        CIFAR10C(corruption_type, corruption_level, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)
    valc_loaders.append(valc_loader)


def validate(val_loader, model):
    # switch to evaluate mode
    model.eval()

    outputs_all = np.zeros((len(val_loader.dataset), 11))

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            # compute output
            output = model(input_var)
            output = output.float()
            outputs_all[i*128:(i+1)*128] = output.cpu().numpy()


    return outputs_all




results = {}
outputs_all = validate(val_loader, model)
results[0] = outputs_all

for corruption_level in range(5):
    outputs_all = validate(valc_loaders[corruption_level], model)
    results[corruption_level+1] = outputs_all



with open("data/"+run_name+"/baseline_outputs_"+corruption_type+'.pkl', 'wb') as f:
    pickle.dump(results, f)