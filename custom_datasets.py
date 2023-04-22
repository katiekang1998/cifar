import os
import torch
from torch.utils.data import Dataset
import numpy as np


class CIFAR10C(Dataset):
    def __init__(self, corruption_type, corruption_level, transform=None):
        corrupted_images = np.load("data/CIFAR-10-C/"+corruption_type+".npy")
        self.images = corrupted_images[10000*corruption_level: 10000*(corruption_level+1)]

        labels = np.load("data/CIFAR-10-C/labels.npy").squeeze()
        self.labels = labels[10000*corruption_level: 10000*(corruption_level+1)]

        self.transform = transform


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



if __name__ == "__main__":
    import IPython; IPython.embed()