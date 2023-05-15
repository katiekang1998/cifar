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
from resnet_officehome import Network
from torchvision.datasets import ImageFolder



class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, True)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)



parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

parser.add_argument('--run-name', dest='run_name',
                    type=str, default="")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()
run_name = args.run_name
torch.backends.cudnn.deterministic=True

# train on only env idx 2
test_envs = [0, 1, 3]
datasets_all = OfficeHome("/home/katie/Desktop/DomainBed/domainbed/data/", test_envs)


test_data , train_data = split_dataset(datasets_all[2], int(len(datasets_all[2])*0.2))
val_data0 = datasets_all[0]
val_data1 = datasets_all[1]
val_data3 = datasets_all[3]


test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=32, shuffle=False,
    num_workers=args.workers, pin_memory=True)

val_loader0 = torch.utils.data.DataLoader(
    val_data0,
    batch_size=32, shuffle=False,
    num_workers=args.workers, pin_memory=True)

val_loader1 = torch.utils.data.DataLoader(
    val_data1,
    batch_size=32, shuffle=False,
    num_workers=args.workers, pin_memory=True)

val_loader3 = torch.utils.data.DataLoader(
    val_data3,
    batch_size=32, shuffle=False,
    num_workers=args.workers, pin_memory=True)

val_loaders = [val_loader0, val_loader1, val_loader3]

model = torch.nn.DataParallel(Network(datasets_all.input_shape, datasets_all.num_classes+1))
model.cuda()



checkpoint_name = "data/"+run_name+"/checkpoint.th"
print("=> loading checkpoint '{}'".format(checkpoint_name))
checkpoint = torch.load(checkpoint_name)
start_epoch = checkpoint['epoch']
best_prec1 = checkpoint['best_prec1']
model.load_state_dict(checkpoint['state_dict'])


def validate(val_loader, model):
    # switch to evaluate mode
    model.eval()

    outputs_all = np.zeros((len(val_loader.dataset), datasets_all.num_classes+1))

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            # compute output
            output = model(input_var)
            output = output.float()
            outputs_all[i*32:(i+1)*32] = output.cpu().numpy()
            # print(output.cpu().numpy()[:, :-1].max(axis=-1))
            # print(np.sum(output.cpu().numpy()[:, :-1].argmax(axis=-1) == target.cpu().numpy()))


    return outputs_all



results = {}
outputs_all = validate(test_loader, model)
results[0] = outputs_all

for i in range(3):
    outputs_all = validate(val_loaders[i], model)
    results[i+1] = outputs_all



with open("data/"+run_name+'/outputs.pkl', 'wb') as f:
    pickle.dump(results, f)