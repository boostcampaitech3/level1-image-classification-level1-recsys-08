import torch.nn as nn
import torch.nn.functional as F
from torchvision import models,transforms

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_pre_trained(model_name, feature_extract=False, pretrained = True):
    num_classes = 18
    if model_name == 'densenet169':
        model = models.densenet169(pretrained=pretrained)
        num_ftrs = model.classifier.in_features
        set_parameter_requires_grad(model, feature_extract)
        model.classifier = nn.Linear(num_ftrs, num_classes)
    
    if model_name == 'densenet161':
        model = models.densenet161(pretrained=pretrained)
        num_ftrs = model.classifier.in_features
        set_parameter_requires_grad(model, feature_extract)
        model.classifier = nn.Linear(num_ftrs, num_classes)

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        set_parameter_requires_grad(model, feature_extract)
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        set_parameter_requires_grad(model, feature_extract)
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    return model

class Pretrained_Model(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model = get_pre_trained(model_name)

    def forward(self, x):
        x = self.model.forward(x)
        return x