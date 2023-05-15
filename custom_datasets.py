import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import skimage as sk
from PIL import Image


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

def impulse_noise(x, severity=1):
    c = [.01, .02, .03, .05, .07][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255

class Pascal3d(Dataset):
    def __init__(self, ID, transform=None):
        # super(Pascal3d, self).__init__()
        if ID:
            with open("data/pascal3d/train_data.pkl", "rb") as f:
                data = pickle.load(f)
        else:
            with open("data/pascal3d/val_data.pkl", "rb") as f:
                data = pickle.load(f)
        self.images = np.array(data[0]).reshape((-1, 1, 128, 128))
        if not ID:
            self.images = impulse_noise(self.images, 5)    
        
        self.labels = (np.array(data[1]).reshape((-1, 3))[:, 2]).squeeze()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class SkinLesionPixels(Dataset):
    def __init__(self, data_type, transform=None):
        # super(Pascal3d, self).__init__()
        if data_type=="train":
            with open("data/SkinLesionPixels/images_train.pkl", "rb") as f:
                self.images = pickle.load(f)
            with open("data/SkinLesionPixels/labels_train.pkl", "rb") as f:
                self.labels = pickle.load(f)
        elif data_type=="val":
            with open("data/SkinLesionPixels/images_val.pkl", "rb") as f:
                self.images = pickle.load(f)
            with open("data/SkinLesionPixels/labels_val.pkl", "rb") as f:
                self.labels = pickle.load(f)
        elif data_type=="test":
            with open("data/SkinLesionPixels/images_test.pkl", "rb") as f:
                self.images = pickle.load(f)
            with open("data/SkinLesionPixels/labels_test.pkl", "rb") as f:
                self.labels = pickle.load(f)
        else:
            raise ValueError("data_type must be either train, val, or test")
        
        assert(len(self.images)==len(self.labels))
        # import IPython; IPython.embed()
        # self.images = np.array(data[0]).reshape((-1, 1, 128, 128))
        # if not ID:
        #     self.images = impulse_noise(self.images, 5)    
        
        # self.labels = (np.array(data[1]).reshape((-1, 3))[:, 2]).squeeze()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        # import IPython; IPython.embed()
        return image, label

class Yearbook(Dataset):
    def __init__(self, ID, transform=None):
        # super(Pascal3d, self).__init__()
        self.ID = ID
        if ID:
            self.image_path = "data/yearbook/faces_aligned_small_mirrored_co_aligned_cropped_cleaned/F"
            
  
        else:
            self.image_path = "data/yearbook/faces_aligned_small_mirrored_co_aligned_cropped_cleaned/M"

        self.image_names = os.listdir(self.image_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        
        image = np.array(sk.io.imread(self.image_path + "/" + image_name)) #Image.open(self.image_path + "/" + image_name) #
        # image = np.transpose(image, (2, 0, 1))
        label = float(image_name.split("_")[0])

        if not self.ID:
            image = impulse_noise(image, 5)

        if self.transform:
            image = self.transform(image)
        return image, (label-1960)/40 #int(label-1905) #


if __name__ == "__main__":
    import IPython; IPython.embed()