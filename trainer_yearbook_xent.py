import torchvision.transforms as transforms
from custom_datasets import Pascal3d, Yearbook

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
from vgg import VGGYearbookXent
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

def get_accuracy(output, target):
    return torch.sum(output.argmax(dim=1) == target).item() / len(target)


def main():
    global args, best_prec1
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic=True

    # run = wandb.init(project="pascal3d", name=args.save_dir)
    run = wandb.init(project="yearbook", name=args.save_dir)
    wandb.config.update(args)


    # Check the save_dir exists or not
    if not os.path.exists(os.path.join("data", args.save_dir)):
        os.makedirs(os.path.join("data", args.save_dir))

    # model = torch.nn.DataParallel(ResNet18(num_classes=2, num_channels=3))
    model = (VGGYearbookXent())
    model.cuda()

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.4660, 0.4660, 0.4660], std=[0.2850, 0.2850, 0.2850])
    # Get the training set
    train_data_all = Yearbook(True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            normalize,
        ]))
    train_data, test_data = torch.utils.data.random_split(train_data_all, [0.9, 0.1]) #train_data_all.split_dataset(0.9)

    val_data_all = Yearbook(False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    
    _, val_data = torch.utils.data.random_split(val_data_all, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=64, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=64, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=64, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    import IPython; IPython.embed()


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    criterion2 = get_accuracy

    
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                weight_decay=0.0)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    best_loss = validate(test_loader, model, criterion, criterion2, "test")
    validate(val_loader, model, criterion, criterion2, "val2")

    for epoch in range(0, 200):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, criterion2, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        loss_curr = validate(test_loader, model, criterion, criterion2, "test")
        validate(val_loader, model, criterion, criterion2, "val2")

        # for corruption_level in range(5):
        #     print(corruption_level+1)
        #     validate(valc_loaders[corruption_level], model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = loss_curr < best_loss
        best_loss = min(loss_curr, best_loss)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_loss,
            }, is_best, filename=os.path.join("data", args.save_dir))


def train(train_loader, model, criterion, criterion2, optimizer, epoch):
    """
        Run one train epoch 
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    vars = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.cuda().float()

        # compute output
        output = model(input)

        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        try:
            losses.update(loss.item(), input.size(0))  
        except:
            import IPython; IPython.embed()      

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, var=vars))

            wandb.log({f'train/loss': float(losses.avg)}, step=epoch*len(train_loader)+i)


def validate(val_loader, model, criterion, criterion2, label):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    mses = AverageMeter()
    vars = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda().float()

            # compute output
            output = model(input_var)
            
            loss = criterion(output, target)
            mse = criterion2(output, target)
            
            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            mses.update(mse, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(label)
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Accuracy {mse.val:.4f} ({mse.avg:.4f})\t'.format(
                  i, len(val_loader), batch_time=batch_time, loss=losses, mse=mses))

        wandb.log({label+'/loss': float(losses.avg)})
        wandb.log({label+'/mse': float(mses.avg)})
        wandb.log({label+'/avg_var': float(vars.avg)})

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
