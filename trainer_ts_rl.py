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
from trainer_rl import get_rl_loss

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

parser.add_argument('--run-name-prefix', dest='run_name_prefix',
                    type=str, default="")
parser.add_argument('--misspecification-cost', dest='misspecification_cost',
                    type=int, default=1)

args = parser.parse_args()

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

# import torch
from torch import nn, optim
from torch.nn import functional as F


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, models):
        super(ModelWithTemperature, self).__init__()
        self.models = models
        self.temperature = nn.Parameter(torch.zeros(11))

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        # temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        # import IPython; IPython.embed()
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), 11)
        return logits + temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        rl_criterion = get_rl_loss(args.misspecification_cost, 0)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = []
                for model in self.models:
                    logits_i = model(input)
                    if len(logits)==0:
                        logits = logits_i.unsqueeze(axis=0)
                    else:
                        logits = torch.cat([logits, logits_i.unsqueeze(axis=0)], axis=0)
                logits = logits.mean(axis=0)

                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = rl_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f' % (before_temperature_nll))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = rl_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = rl_criterion(self.temperature_scale(logits), labels).item()
        # print('Optimal temperature: %.3f' % self.temperature)
        print(self.temperature)
        print('After temperature - NLL: %.3f' % (after_temperature_nll))

        return self

ModelWithTemperature(models).set_temperature(val_loader)