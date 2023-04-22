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

# import torch
from torch import nn, optim
from torch.nn import functional as F

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

parser.add_argument('--run-name-prefix', dest='run_name_prefix',
                    type=str, default="")
parser.add_argument('--corruption-type', dest='corruption_type',
                    help='Type of corruption to add to evaluation images',
                    type=str, default="")
parser.add_argument('--corruption-level', dest='corruption_level',
                    type=int, default=0)
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

if args.corruption_type == "":
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)
else:
    val_loader = torch.utils.data.DataLoader(
        CIFAR10C(args.corruption_type, args.corruption_level, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)



class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

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
        self.temperature = nn.Parameter(torch.ones(10) * 1.5)

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
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), 10)
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()


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
        before_temperature_nll = ece_criterion(logits, labels).item()
        print('Before temperature - ECE: %.3f' % (before_temperature_nll))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=500)

        def eval():
            optimizer.zero_grad()
            loss = ece_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = ece_criterion(self.temperature_scale(logits), labels).item()
        # print('Optimal temperature: %.3f' % self.temperature)
        print(self.temperature)
        print('After temperature - ECE: %.3f' % (after_temperature_nll))

        return self.temperature

best_temperature = ModelWithTemperature(models).set_temperature(val_loader)

if not os.path.exists("data/"+args.run_name_prefix+"_ensemble_eval/"):
    os.makedirs("data/"+args.run_name_prefix+"_ensemble_eval/")

if args.corruption_type == "":
    np.save("data/"+args.run_name_prefix+"_ensemble_eval/best_temp.npy", best_temperature.detach().cpu().numpy())
else:
    np.save("data/"+args.run_name_prefix+"_ensemble_eval/best_temp_"+args.corruption_type+"_"+str(args.corruption_level)+".npy", best_temperature.detach().cpu().numpy())

