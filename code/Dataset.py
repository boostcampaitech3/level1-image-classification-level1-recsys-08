import os
import random
import collections
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from typing import Tuple

class ImageDataset(Dataset):
    '''
    csv 데이터를 통해 만들어진 Dataset Class
    input: base(string)
           filename(string)
           transform(torchvision.transforms.transforms, default=None)
           train(bool, default=True)
    '''
    
    def __init__(self, base, filename, transform=None):
        self.data = pd.read_csv(base+'/'+filename)
        self.transform = transform
        self.path = base
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = Image.open(self.path+'/'+self.data['image_path'][idx])
        y = self.data['class'][idx]
        
        if self.transform:
            X = self.transform(X)
            
        return X, y
