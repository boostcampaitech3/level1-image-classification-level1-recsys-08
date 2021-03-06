import math

import torch
import torch.nn as nn
import torchvision


class ResNet18(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.in_features = 512

        if 'in_features' in kwargs:
            self.in_features = kwargs.get('in_features')

        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18.fc = torch.nn.Linear(in_features=self.in_features, out_features=self.num_classes, bias=True)

        torch.nn.init.kaiming_normal_(self.resnet18.fc.weight)
        stdv = 1. / math.sqrt(self.resnet18.fc.weight.size(1))
        self.resnet18.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.resnet18(x)
        return x