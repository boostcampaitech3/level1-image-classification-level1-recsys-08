from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
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