from torchvision.transforms import InterpolationMode, Resize, ToTensor, Normalize, Compose


class BaseAugmentation:
    def __init__(self, resize, mean, std, **kwargs):
        self.transform = Compose([
            Resize(resize, InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)