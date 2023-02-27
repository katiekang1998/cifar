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

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

parser.add_argument('--run-name', dest='run_name',
                    type=str, default="")
parser.add_argument('--corruption-type', dest='corruption_type',
                    help='Type of corruption to add to evaluation images',
                    type=str, default="impulse_noise")
parser.add_argument('--misspecification-cost', dest='misspecification_cost',
                    type=int, default=1)
args = parser.parse_args()
run_name = args.run_name
corruption_type = args.corruption_type
misspecification_cost = args.misspecification_cost

xent = "xent" in run_name
checkpoint_name = run_name+"/checkpoint.th"

if xent:
    model = torch.nn.DataParallel(resnet.__dict__['resnet20']())
else:
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


def validate(val_loader, model, use_threshold):
    threshold = misspecification_cost/(1+misspecification_cost)
    # switch to evaluate mode
    model.eval()

    reward = AverageMeter()
    accuracy = AverageMeter()
    a10_ratio = AverageMeter()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()
            # compute output
            output = model(input_var)
            output = output.float()

            one_hot = nn.functional.one_hot(target_var.to(torch.int64), 11)
            reward_all = (misspecification_cost+1)*one_hot - misspecification_cost
            reward_all[:, -1] = 0
            output_dist_max, best_actions = nn.functional.softmax(output, dim=-1).max(axis=-1)
            accuracy.update((best_actions==target).sum()/len(target), input.size(0))

            if use_threshold:
                certain = (output_dist_max > threshold).to(torch.int64).cuda()
                actions = certain*best_actions + (1-certain)*torch.ones(best_actions.shape).cuda()*10
                a10_ratio.update((1-certain).sum()/len(certain), len(certain))
            else:
                actions = best_actions
                a10_ratio.update((actions==10).sum()/len(actions), len(actions))
            

            reward.update(torch.mean(torch.gather(reward_all, -1, actions.to(torch.int64).unsqueeze(-1)).type(torch.DoubleTensor)), input.size(0))

    return reward.avg.item(), accuracy.avg.item(), a10_ratio.avg.item()



results = {}
results["reward"] = np.zeros(6)
results["accuracy"] = np.zeros(6)
results["a10_ratio"] = np.zeros(6)


print(xent)

results["reward"][0], results["accuracy"][0], results["a10_ratio"][0] = validate(val_loader, model, xent)

for _ in range(5):
    results["reward"][_+1], results["accuracy"][_+1], results["a10_ratio"][_+1] = validate(valc_loaders[_], model, xent)

print(results)

import pickle


with open(run_name+"/"+corruption_type+'_mc'+str(misspecification_cost)+'.pkl', 'wb') as f:
    pickle.dump(results, f)