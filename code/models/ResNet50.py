import math

import torch
import torch.nn as nn
import torchvision


class ResNet50(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.in_features = 512

        if 'in_features' in kwargs:
            self.in_features = kwargs.get('in_features')

        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.fc = torch.nn.Linear(in_features=self.in_features, out_features=self.num_classes, bias=True)

        torch.nn.init.kaiming_normal_(self.resnet50.fc.weight)
        stdv = 1. / math.sqrt(self.resnet50.fc.weight.size(1))
        self.resnet50.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.resnet50(x)
        return x