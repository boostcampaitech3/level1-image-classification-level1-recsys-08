import os
import sys
import random
from collections import defaultdict
from typing import List

from torch.utils.data import Subset

module_path = os.path.dirname(os.path.realpath(__file__))
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)

from BaseDataset import BaseDataset


class SplitByProfileDataset(BaseDataset):

    def __init__(self, data_dir,
                 val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self.valid_file_name:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile,
                                            file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self.valid_file_name[_file_name]

                    _, gender, race, age = profile.split("_")
                    gender_label = self.GenderLabels.get_label(gender)
                    age_label = self.AgeLabels.get_label(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]
