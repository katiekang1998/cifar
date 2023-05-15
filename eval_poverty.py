from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms

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
from resnet_ms import ResNet18
import wandb
import numpy as np
import random
import pickle


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

parser.add_argument('--run-name', dest='run_name',
                    type=str, default="")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()
run_name = args.run_name
torch.backends.cudnn.deterministic=True

dataset = get_dataset(dataset="poverty", download=False)

# Get the training set
train_data = dataset.get_subset(
    "train",
    # transform=transforms.Compose(
    # [transforms.ToTensor()]
# ),
)


test_data = dataset.get_subset(
    "id_val")


# Get the validation set
val_data1 = dataset.get_subset(
    "val",
#     transform=transforms.Compose(
#     [transforms.ToTensor()]
# ),
)

# Get the test set
val_data2 = dataset.get_subset(
    "test",
#     transform=transforms.Compose(
#     [transforms.ToTensor()]
# ),
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=64, shuffle=False,
    num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=64, shuffle=False,
    num_workers=4, pin_memory=True)

val_loader1 = torch.utils.data.DataLoader(
    val_data1,
    batch_size=64, shuffle=False,
    num_workers=4, pin_memory=True)

val_loader2 = torch.utils.data.DataLoader(
    val_data2,
    batch_size=64, shuffle=False,
    num_workers=4, pin_memory=True)


model = torch.nn.DataParallel(ResNet18(num_classes=2, num_channels=8))
model.cuda()

checkpoint_name = "data/"+run_name+"/best.th"

print("=> loading checkpoint '{}'".format(checkpoint_name))
checkpoint = torch.load(checkpoint_name)
model.load_state_dict(checkpoint['state_dict'])

model.eval()


def validate(val_loader, model):
    # switch to evaluate mode
    model.eval()

    outputs_all = np.zeros((len(val_loader.dataset), 3))

    with torch.no_grad():
        for i, (input, target, metadata) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda().squeeze()

            output = model(input)


            mean = output[:, 0]
            log_var = torch.clip(output[:, 1], -20, 2)
            var = torch.exp(log_var)
            outputs_all[i*64:(i+1)*64, 0] = mean.cpu().numpy()
            outputs_all[i*64:(i+1)*64, 1] = np.sqrt(var.cpu().numpy())
            outputs_all[i*64:(i+1)*64, 2] = ((mean-target)**2).cpu().numpy()
    


    return outputs_all



results = {}
outputs_all = validate(test_loader, model)
results["train"] = outputs_all


test_countries = ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania', 'angola', 'cote_d_ivoire', 'ethiopia', 'mali', 'rwanda']


for country in test_countries:
    country_metadata_idx = val_data1._metadata_map["country"].index(country)
    country_idxs = np.where(val_data1.dataset._metadata_array[:, 2] == country_metadata_idx)
    country_data = torch.utils.data.Subset(dataset, country_idxs[0])
    country_loader = torch.utils.data.DataLoader(
        country_data,
        batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True)
    outputs_all = validate(country_loader, model)
    results[country] = outputs_all



with open("data/"+run_name+'/outputs.pkl', 'wb') as f:
    pickle.dump(results, f)