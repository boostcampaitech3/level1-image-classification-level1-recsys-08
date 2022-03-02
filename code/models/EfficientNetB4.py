import torch.nn as nn
import torchvision.models as models


class EfficientNetB4(nn.Module):
    def __init__(self, num_classes: int = 18, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.efficientnet_b4(pretrained=self.pretrained)
        self.in_features = self.model.classifier[1].out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x
