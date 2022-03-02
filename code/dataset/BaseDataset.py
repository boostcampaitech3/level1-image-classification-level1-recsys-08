import os
import re

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms


class BaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    class MaskLabels:
        MASK = 0
        INCORRECT = 1
        NORMAL = 2

    class GenderLabels:
        MALE = 0
        FEMALE = 1

        @classmethod
        def get_label(cls, value: str):
            gender = getattr(BaseDataset.GenderLabels, value.upper())
            try:
                return gender
            except Exception:
                raise ValueError(f"Gender value should be either 'male' or 'female', {value}")

    class AgeLabels:
        YOUNG = 0
        MIDDLE = 1
        OLD = 2

        @classmethod
        def get_label(cls, value: str):
            try:
                value = int(value)
            except Exception:
                raise ValueError(f"Age value should be numeric, {value}")

            if value < 30:
                return cls.YOUNG
            elif value < 60:
                return cls.MIDDLE
            else:
                return cls.OLD

    img_extensions = [
        ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"
    ]

    valid_file_name = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    # 클래스의 정적 메소드를 정의합니다.

    @staticmethod
    def is_image(file):
        # 파일의 확장자만을 분리하여 저장합니다.
        file_name, file_extension = os.path.splitext(file)

        # 파일의 확장자가 이미지인지 아닌지 여부를 반환합니다.
        return (file_extension in BaseDataset.img_extensions)

    @staticmethod
    def encode_multi_class(age_label, gender_label, mask_label):
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(class_label):
        mask_label = (class_label // 6) % 3
        gender_label = (class_label // 3) % 2
        age_label = (class_label % 3)
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    # 데이터 셋 모델 인스턴스를 초기에 생성할 때 실행해야 하는 메소드를 정의합니다.

    def __init__(self, data_dir,
                 val_ratio:float=0.2,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 transform=None):

        self.data_dir = data_dir
        self.mean = mean
        self.std = std

        # 데이터의 feature value를 저장합니다.
        self.image_paths = list()
        self.age_labels = list()
        self.gender_labels = list()
        self.mask_labels = list()

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.set_transform(transform)

        self.val_ratio = val_ratio
        self.set_up()

    def set_up(self):
        data_dir = self.data_dir
        dir_list = os.listdir(data_dir)

        for directory in dir_list:
            # "."로 시작하는 파일 및 폴더는 무시합니다.
            if re.match('^[.]', directory):
                continue

            image_dir = os.path.join(data_dir, directory)
            for image in os.listdir(image_dir):
                # 이미지가 맞는지 아닌지 확인합니다.
                if not self.is_image(image):
                    continue

                image_name, _ = os.path.splitext(image)
                # 이미지 파일명이 유효한지 확인합니다.
                if image_name not in self.valid_file_name:
                    continue

                # 데이터 별로 존재하는 이미지 폴더명을 분리하여 데이터를 얻습니다.
                id, gender, race, age = directory.split('_')
                image_path = os.path.join(image_dir, image)
                age_label = self.AgeLabels.get_label(age)
                gender_label = self.GenderLabels.get_label(gender)
                mask_label = self.valid_file_name[image_name]

                self.image_paths.append(image_path)
                self.age_labels.append(age_label)
                self.gender_labels.append(gender_label)
                self.mask_labels.append(mask_label)

    def set_transform(self, transform):
        self.transform = transform


    # 필요한 데이터 label을 가져오는 데 필요한 메소드를 정의합니다.

    def __getitem__(self, index):
        image = self.get_image(index)
        age_label = self.get_age_label(index)
        gender_label = self.get_gender_label(index)
        mask_label = self.get_mask_label(index)
        label = self.encode_multi_class(age_label, gender_label, mask_label)

        transformed_image = self.transform(image)
        return transformed_image, label

    def get_image(self, index):
        return Image.open(self.image_paths[index])

    def get_age_label(self, index):
        return self.age_labels[index]

    def get_gender_label(self, index):
        return self.gender_labels[index]

    def get_mask_label(self, index):
        return self.mask_labels[index]

    # 데이터 셋의 크기를 반환하는 메소드를 정의합니다.

    def __len__(self):
        return len(self.image_paths)

    # 데이터 index에 대응되는 데이터를 반환합니다.

    def get_age(self, index):
        age_label = self.get_age_label(index)
        age = ['YOUNG', 'MIDDLE', 'OLD'][age_label]
        return age

    def get_gender(self, index):
        gender_label = self.get_gender_label(index)
        gender = ['MALE', 'FEMALE'][gender_label]
        return gender

    def get_mask(self, index):
        mask_label = self.get_mask_label(index)
        mask = ['WEAR', 'INCORRECT', 'NOT WEAR'][mask_label]
        return mask

    def split_dataset(self):
        data_size = len(self)
        val_set_size = int(data_size * self.val_ratio)
        train_set_size = data_size - val_set_size
        train_set, val_set = random_split(self, [train_set_size, val_set_size])
        return train_set, val_set


