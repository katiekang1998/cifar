from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

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
from resnet_officehome import Network
import wandb
import numpy as np
import random


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--seed', dest='seed', type=int, default=48)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=1)

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

def main():
    global args, best_prec1
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic=True

    run = wandb.init(project="officehome", name=args.save_dir)
    wandb.config.update(args)


    # Check the save_dir exists or not
    if not os.path.exists(os.path.join("data", args.save_dir)):
        os.makedirs(os.path.join("data", args.save_dir))



    cudnn.benchmark = True


    # train on only env idx 2
    test_envs = [0, 1, 3]
    datasets_all = OfficeHome("/home/katie/Desktop/DomainBed/domainbed/data/", test_envs)


    test_data , train_data = split_dataset(datasets_all[2], int(len(datasets_all[2])*0.2))

    val_data0 = datasets_all[0]
    val_data1 = datasets_all[1]
    val_data3 = datasets_all[3]

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=32, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=32, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader0 = torch.utils.data.DataLoader(
        val_data0,
        batch_size=32, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader1 = torch.utils.data.DataLoader(
        val_data1,
        batch_size=32, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader3 = torch.utils.data.DataLoader(
        val_data3,
        batch_size=32, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    model = torch.nn.DataParallel(Network(datasets_all.input_shape, datasets_all.num_classes))
    model.cuda()
    
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=5e-05,
            weight_decay=0
        )


    best_loss = 1000000

    for epoch in range(0, 80):

        # train for one epoch
        train(train_loader, model, optimizer, epoch)

        # evaluate on validation set
        loss_curr = validate(test_loader, model, "test")
        validate(val_loader0, model, "val0")
        validate(val_loader1, model, "val1")
        validate(val_loader3, model, "val3")

        # remember best prec@1 and save checkpoint
        is_best = loss_curr < best_loss
        best_loss = min(loss_curr, best_loss)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_loss,
            }, is_best, filename=os.path.join("data", args.save_dir))


def train(train_loader, model, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    entropies = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda().squeeze()
        input = input.cuda()

        # compute output
        output = model(input)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output, target)

        # #take softmax
        distribution = torch.nn.functional.softmax(output, dim=1)

        entropy = -torch.sum(distribution * torch.log(torch.clip(distribution, min=0.00001)), dim=1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        entropies.update(entropy.mean().item(), input.size(0))
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Average entropy {ent.val:.4f} ({ent.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, ent=entropies))

            wandb.log({f'train/loss': float(losses.avg)}, step=epoch*len(train_loader)+i)
            wandb.log({f'train/avg_ent': float(entropies.avg)}, step=epoch*len(train_loader)+i)


def validate(val_loader, model, label):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    entropies = AverageMeter()
    accuracies = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda().squeeze()
            input_var = input.cuda()

            # compute output
            output = model(input_var)

            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(output, target)

            distribution = torch.nn.functional.softmax(output, dim=1)

            entropy = -torch.sum(distribution * torch.log(torch.clip(distribution, min=0.00001)), dim=1)
            acc = output.argmax(dim=1).eq(target).sum().item() / target.size(0)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            entropies.update(entropy.mean().item(), input.size(0))
            accuracies.update(acc, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(label)
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Average entropy {ent.val:.4f} ({ent.avg:.4f})\t'
              'Accuarcy {acc.val:.4f} ({acc.avg:.4f}'.format(
                  i, len(val_loader), batch_time=batch_time, loss=losses, ent=entropies, acc=accuracies))

        wandb.log({label+'/loss': float(losses.avg)})
        wandb.log({label+'/avg_ent': float(entropies.avg)})
        wandb.log({label+'/acc': float(accuracies.avg)})

    # print(' * Prec@1 {top1.avg:.3f}'
    #       .format(top1=top1))

    return losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, os.path.join(filename,  'checkpoint.th'))
    if is_best:
        torch.save(state, os.path.join(filename,  'best.th'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
