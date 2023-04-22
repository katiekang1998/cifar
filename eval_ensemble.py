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

parser.add_argument('--run-name-prefix', dest='run_name_prefix',
                    type=str, default="")
parser.add_argument('--corruption-type', dest='corruption_type',
                    help='Type of corruption to add to evaluation images',
                    type=str, default="impulse_noise")
parser.add_argument('--misspecification-cost', dest='misspecification_cost',
                    type=int, default=1)
parser.add_argument('--ts', dest='ts',
                    type=bool, default=False)
parser.add_argument('--ts-oracle', dest='ts_oracle',
                    type=bool, default=False)
parser.add_argument('--oracle-threshold', dest='oracle_threshold',
                    type=bool, default=False)
parser.add_argument('--dont-use-threshold', dest='dont_use_threshold',
                    type=bool, default=False)
args = parser.parse_args()

corruption_type = args.corruption_type
misspecification_cost = args.misspecification_cost
use_threshold = not args.dont_use_threshold

xent = "xent" in args.run_name_prefix


run_names = [f for f in os.listdir("data") if args.run_name_prefix+"_seed" in f]

models = []
for run_name in run_names:
    checkpoint_name = "data/"+run_name+"/checkpoint.th"

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
    models.append(model)

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


def validate(val_loader, models, use_threshold, temperature, threshold=None):
    if threshold == None:
        threshold = misspecification_cost/(1+misspecification_cost)
    # switch to evaluate mode

    for model in models:
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

            output = []
            for model in models:
                output_i = model(input_var)
                if len(output)==0:
                    output = output_i.unsqueeze(axis=0)
                else:
                    output = torch.cat([output, output_i.unsqueeze(axis=0)], axis=0)
            output = output.mean(axis=0)
            output = output.float()
            if use_threshold and (args.ts or args.ts_oracle):
                output = output/torch.Tensor(np.expand_dims(temperature, axis=0)).cuda()

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


if args.oracle_threshold:
    # ts = np.arange(misspecification_cost/(1+misspecification_cost), 1.01, 0.01)
    ts = np.arange(0.95, 1.001, 0.001)
    best_r = -100
    best_t = 0
    for t in (ts):
        reward, accuracy, ratio = validate(val_loader, models, xent, None, threshold = t)
        if reward>best_r:
            results["reward"][0], results["accuracy"][0], results["a10_ratio"][0]  = reward, accuracy, ratio
            best_t = t
            best_r = reward
    print(best_t)
    print(best_r)

    for corruption_level in range(5):
        best_r = -100
        best_t = 0
        for t in (ts):
            reward, accuracy, ratio = validate(valc_loaders[corruption_level], models, xent, None, threshold = t)
            if reward>best_r:
                results["reward"][corruption_level+1], results["accuracy"][corruption_level+1], results["a10_ratio"][corruption_level+1]  = reward, accuracy, ratio
                best_t = t 
                best_r = reward
        print(best_t)
        print(best_r)

else:
    temperature = None
    if args.ts or args.ts_oracle:
        temperature = np.load("data/"+args.run_name_prefix+"_ensemble_eval/best_temp.npy")
    results["reward"][0], results["accuracy"][0], results["a10_ratio"][0] = validate(val_loader, models, xent and use_threshold, temperature)

    for corruption_level in range(5):
        if args.ts_oracle:
            temperature = np.load("data/"+args.run_name_prefix+"_ensemble_eval/best_temp_"+args.corruption_type+"_"+str(corruption_level)+".npy")
        results["reward"][corruption_level+1], results["accuracy"][corruption_level+1], results["a10_ratio"][corruption_level+1] = validate(valc_loaders[corruption_level], models, xent and use_threshold, temperature)

print(results)

import pickle

if not os.path.exists("data/"+args.run_name_prefix+"_ensemble_eval/"):
    os.makedirs("data/"+args.run_name_prefix+"_ensemble_eval/")

if args.oracle_threshold:
    postfix = '_oracle_threshold'
elif args.ts:
    postfix = '_ts'
elif args.ts_oracle:
    postfix = '_ts_oracle'
elif args.dont_use_threshold:
    postfix = '_no_threshold'
else:
    postfix = ""

print(postfix)
with open("data/"+args.run_name_prefix+"_ensemble_eval/"+corruption_type+'_mc'+str(misspecification_cost)+postfix+'.pkl', 'wb') as f:
    pickle.dump(results, f)