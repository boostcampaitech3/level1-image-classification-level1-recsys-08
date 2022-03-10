from torchvision.transforms import *


class TrainAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            RandomChoice([
                CenterCrop((384, 384)),
                ColorJitter(brightness=(0.2, 3)),
                Grayscale(num_output_channels=3),
            ]),
            Resize(resize, InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.2),
            RandomPerspective(distortion_scale=0.2, p=0.2),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)