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

model = torch.nn.DataParallel(resnet.__dict__['resnet20']())
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


def validate(val_loader, model, temperature):
    # switch to evaluate mode
    model.eval()

    outputs_all = np.zeros((len(val_loader.dataset), 2))
    #return, best predicted action

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            # compute output
            output = model(input_var)
            output = output.float()
            output = output/torch.from_numpy(np.expand_dims(temperature, axis=0)).cuda()
            # else:
            #     prob = ((output[:, :-1]+misspecification_cost)/(1+misspecification_cost)).clip(min=0.00000001)
            #     prob_ts = nn.functional.softmax(torch.log(prob)/args.temperature, dim=-1)
            #     output[:, :-1] = (1+misspecification_cost)*prob_ts- misspecification_cost

            one_hot = nn.functional.one_hot(target.to(torch.int64), 11)
            reward_all = (4+1)*one_hot - 4
            reward_all[:, -1] = 0
            output_dist_max, best_actions = nn.functional.softmax(output, dim=-1).max(axis=-1)
            threshold = 4/(1+4)
            certain = (output_dist_max > threshold).to(torch.int64).cuda()
            actions = certain*best_actions + (1-certain)*torch.ones(best_actions.shape).cuda()*10

            reward = torch.gather(reward_all, -1, actions.to(torch.int64).unsqueeze(-1)).type(torch.DoubleTensor).squeeze()
            outputs_all[i*128:(i+1)*128, 0] = reward.cpu().numpy()
            outputs_all[i*128:(i+1)*128, 1] = actions.cpu().numpy()
    print(outputs_all[:, 0].mean())
    return outputs_all




results = {}
temperature = np.load("data/"+run_name+"/best_temp.npy")
outputs_all = validate(val_loader, model, temperature)
results[0] = outputs_all

for corruption_level in range(5):
    temperature = np.load("data/"+run_name+"/best_temp_"+corruption_type+"_"+str(corruption_level)+".npy")
    outputs_all = validate(valc_loaders[corruption_level], model, temperature)
    results[corruption_level+1] = outputs_all



with open("data/"+run_name+"/oracle_outputs_"+corruption_type+'.pkl', 'wb') as f:
    pickle.dump(results, f)