

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGGYearbook(nn.Module):
    def __init__(self):
        super(VGGYearbook, self).__init__()
        model = models.vgg16(pretrained=True)

        self.features = model.features
        self.avgpool = model.avgpool
        self.fc = nn.Sequential(nn.Linear(25088, 4096), nn.ReLU(inplace=True), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Linear(4096, 2))



    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class VGGYearbook2(nn.Module):
    def __init__(self):
        super(VGGYearbook2, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.linear = nn.Linear(1000, 2)


    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x

class VGGYearbookXent(nn.Module):
    def __init__(self):
        super(VGGYearbookXent, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.linear = nn.Linear(1000, 109)


    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x
