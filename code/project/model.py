import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch import EfficientNet


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = torchvision.models.resnet18(pretrained=True)
        self.net.fc = nn.Linear(512, num_classes, bias=True)
        self.freeze_parameters()
        self.initialization()

    def freeze_parameters(self):
        for name, param in self.net.named_parameters():
            if name.startswith('fc'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def initialization(self):
        torch.nn.init.xavier_uniform_(self.net.fc.weight)
        stdv = 1. / math.sqrt(self.net.fc.weight.size(1))
        self.net.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        out = self.net(x)
        return out


class EfficientModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientModel, self).__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

    def forward(self, x):
        out = self.net(x)
        return out