import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop, ColorJitter, RandomHorizontalFlip

class TrainAugmentation:
    def __init__(self, **args):
        self.transform = Compose([
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __call__(self, image):
        return self.transform(image)

class EvalAugmentation:
    def __init__(self, **args):
        self.transform = Compose([
            # Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __call__(self, image):
        return self.transform(image)