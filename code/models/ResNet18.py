import math

import torch
import torch.nn as nn
import torchvision


class ResNet18(nn.Module):
    def __init__(self, num_classes: int, in_features: int = 512):
        super().__init__()

        self.num_classes = num_classes
        self.in_features = in_features

        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18.fc = torch.nn.Linear(in_features=self.in_features, out_features=self.num_classes, bias=True)

        torch.nn.init.kaiming_normal_(self.resnet18.fc.weight)
        stdv = 1. / math.sqrt(self.resnet18.fc.weight.size(1))
        self.resnet18.fc.bias.data.uniform_(-stdv, stdv)

        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.resnet18(x)
        x = self.layer(x)
        return x