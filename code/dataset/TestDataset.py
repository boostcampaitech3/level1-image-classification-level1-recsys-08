import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TestDataset(Dataset):
    def __init__(self,
                 test_data_dir,
                 test_data_file,
                 transform=None,
                 mean=(0.548, 0.504, 0.479),
                 std=(0.237, 0.247, 0.246)):
        self.info = pd.read_csv(test_data_file)
        self.mean = mean
        self.std = std

        img_paths = [os.path.join(test_data_dir, img_id) for img_id in self.info.ImageID]
        self.img_paths = img_paths
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.set_transform(transform)

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

    def set_transform(self, transform):
        self.transform = transform