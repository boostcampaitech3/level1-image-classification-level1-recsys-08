import timm
import torch.nn as nn


class ResNet152(nn.Module):
    def __init__(self, model_arch='tv_resnet152', n_class=18, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        return self.model(x)