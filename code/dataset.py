from torch.utils.data import DataLoader,Dataset, Subset, random_split
from torchvision import models,transforms
import pandas as pd
import numpy as np
import os
from PIL import Image
from typing import Tuple, List
import torch

data_dir = '../../input/data/train'

def label_decoder(n:int):
    d = dict()
    keys = [(i,j,k) for i in range(3) for j in range(2) for k in range(3)]
    labels = range(18)
    for x,y in zip(keys,labels):
        d[y] = x
    result = (['Wear','Incorrect','Not Wear'][d[n][0]], ['Male','Female'][d[n][1]], ['<30','30<= & <60','60<='][d[n][2]])
    return result

def label_encoder(n:int):
    d = dict()
    keys = [(i,j,k) for i in range(3) for j in range(2) for k in range(3)]
    labels = range(18)
    for x,y in zip(keys,labels):
        d[x] = y
    
    return d[n]

class MaskDataset(Dataset):
        
        def __init__(self, data_dir: str, transforms=None, adj_csv:bool =True, val_ratio: float = 0.2, up_sampling:int = 0, augment:int = 0):
            '''
            Agrs:
                data_dir (str): 데이터 경로
                adj_csv (bool): 데이터 경로에 라벨링 완료된 train_adj.csv 존재 여부
                up_sampling (int): 30대, 40대, 60세의 데이터를 up_sampling배만큼 중복 추가
            
            '''
            self.data_dir = data_dir
            self.df_train = pd.read_csv(self.data_dir + '/' + 'train.csv')
            self.image_dir = self.data_dir + '/images'
            self.transforms = transforms
            self.val_ratio = val_ratio
            
            self.id = []
            self.gender_labels = []
            self.age_labels = []
            self.path = []
            self.mask_labels = []
            self.extension = []
            
            # train_adj 없는 경우 생성, 있는 경우 불러옴
            if not adj_csv:
                for i in range(len(self.df_train)):
                    tmp = self.df_train.iloc[i]
                    id_ = tmp['id']
                    gender_ = int(tmp['gender']=='female')
                    age_ = tmp['age']
                    path1 = tmp["path"]
                    for path2 in os.listdir(self.image_dir + '/' + path1):
                        if path2.startswith("._"):
                            continue
                        name, ext = path2.split('.')
                        for k, x in enumerate(['mask','incorrect','normal']):
                            if name.startswith(x):
                                self.mask_labels.append(k)
                        self.path.append(os.path.join(self.image_dir, path1 + '/' + path2))
                        self.id.append(id_)
                        self.gender_labels.append(gender_)
                        self.age_labels.append(age_)
                        self.extension.append(ext)

                self.df_train_adj = pd.DataFrame(data = zip(self.id,self.path,self.extension,self.age_labels,self.mask_labels,self.gender_labels), 
                                                 columns = ['id','path','extension','age','mask','gender'])
                # (30세 미만, 30세 이상 60세 미만, 60세 이상) = (0,1,2)
                self.df_train_adj['age_class'] = pd.cut(self.df_train_adj['age'], bins = [0,30,58,1000], right=False, labels = [0,1,2])                             
                # 정답 레이블 컬럼 추가
                self.df_train_adj['ans'] = [self.class_dict[tuple(self.df_train_adj[['mask','gender','age_class']].iloc[i])] for i in range(len(self.df_train_adj))] 
                self.df_train_adj.to_csv(self.data_dir + '/' + 'train_adj.csv', index = False)

                self.age_labels = list(self.df_train_adj['age_class'])
                self.image_paths = list(self.df_train_adj['path'])
                self.labels = list(self.df_train_adj['ans'])
            else:
                self.df_train_adj = pd.read_csv(self.data_dir + '/' + 'train_adj.csv')
                self.image_paths = list(self.df_train_adj['path'])
                self.labels = list(self.df_train_adj['ans'])
                self.age_labels = list(self.df_train_adj['age_class'])
                self.mask_labels = list(self.df_train_adj['mask'])
                self.gender_labels = list(self.df_train_adj['gender'])
                self.age = list(self.df_train_adj['age'])
                
            if up_sampling or augment:
                print('upsamling starts ...')
                cnt = 0
                for i, age in enumerate(self.age):
                    if (30 <= age and age < 50) or (age == 60):
                        for _ in range(up_sampling):
                            self.image_paths.append(self.df_train_adj['path'][i])
                            self.labels.append(self.df_train_adj['ans'][i])
                    elif augment and augment < age < 60:
                        cnt += 1
                        self.image_paths.append(self.df_train_adj['path'][i])
                        a,b,_ = self.class_dict_decode[self.df_train_adj['ans'][i]]
                        self.labels.append(self.class_dict[(a,b,2)])
                if augment:
                    print(f'{augment+1}세 이상 {cnt}명 레이블 2로 추가')

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, index):
            X = Image.open(self.image_paths[index])
            y = self.labels[index]
            if self.transforms:
                X = self.transforms(X)
                
            return X, y
        
        def get_mask_label(self, index) -> 'mask_label':
            tmp = ['Wear', 'Incorrect', 'Not Wear'][self.mask_labels[index]]
            print(f'마스크 착용 여부: {tmp}')
            return self.mask_labels[index]

        def get_gender_label(self, index) -> 'gender_label':
            tmp = ['Male', 'Female'][self.gender_labels[index]]
            print(f'성별: {tmp}')
            return self.gender_labels[index]

        def get_age_label(self, index) -> 'age_label':
            tmp = self.age[index]
            print(f'Age: {tmp}')
            return self.age_labels[index]

        def read_image(self, index) -> 'print image':
            image_path = self.image_paths[index]
            tmp = ['Wear', 'Incorrect', 'Not Wear'][self.mask_labels[index]]
            print(f'마스크 착용 여부: {tmp}')
            tmp = ['Male', 'Female'][self.gender_labels[index]]
            print(f'성별: {tmp}')
            tmp = self.age[index]
            print(f'Age: {tmp}')
            print(f'Class: {self.labels[index]}')
            return Image.open(image_path)
        
        def split_dataset(self) -> Tuple[Subset, Subset]:
            val_ratio = self.val_ratio
            n_val = int(len(self) * val_ratio)
            n_train = len(self) - n_val
            train_set, val_set = random_split(self, [n_train, n_val])
            print(f'Data split completed: {val_ratio=}')
            print(f'{n_train=}, {n_val=}')
            return train_set, val_set
        
class TestDataset(Dataset):
    def __init__(self, img_paths, transforms = None):
        self.img_paths = img_paths
        assert transforms is not None, "transforms를 설정해주세요"
        self.transform = transforms

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
    
class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
class MaskDatasetForSplit(Dataset):
        
        def __init__(self, data_dir: str, transforms=None, adj_csv:bool =True, val_ratio: float = 0.2, up_sampling:int = 0, augment:int = 0, key:str = 'Mask'):
            '''
            Agrs:
                data_dir (str): 데이터 경로
                adj_csv (bool): 데이터 경로에 라벨링 완료된 train_adj.csv 존재 여부
                up_sampling (int): 30대, 40대, 60세의 데이터를 up_sampling배만큼 중복 추가
            
            '''
            self.data_dir = data_dir
            self.df_train = pd.read_csv(self.data_dir + '/' + 'train.csv')
            self.image_dir = self.data_dir + '/images'
            self.transforms = transforms
            self.val_ratio = val_ratio
            self.key = key
            
            self.id = list(self.df_train['id'])
            self.gender = []
            self.age = []
            path = []
            self.mask_labels = []
            self.extension_labels = []
            
            # train_adj 없는 경우 생성, 있는 경우 불러옴
            self.df_train_adj = pd.read_csv(self.data_dir + '/' + 'train_adj.csv')
            self.image_paths = list(self.df_train_adj['path'])
            if self.key == 'Mask':
                self.labels = list(self.df_train_adj['mask'])
            elif self.key == 'Gender':
                self.labels = list(self.df_train_adj['gender'])
            else:
                self.labels = list(self.df_train_adj['age_class'])
            self.age_labels = list(self.df_train_adj['age_class'])
            self.mask_labels = list(self.df_train_adj['mask'])
            self.gender_labels = list(self.df_train_adj['gender'])
            self.age = list(self.df_train_adj['age'])
                

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, index):
            X = Image.open(self.image_paths[index])
            y = self.labels[index]
            if self.transforms:
                X = self.transforms(X)
                
            return X, y
        
        def split_dataset(self, n_fold:int = 0) -> Tuple[Subset, Subset]:
            val_ratio = self.val_ratio
            n_val = int(len(self) * val_ratio)
            n_train = len(self) - n_val
            train_set, val_set = random_split(self, [n_train, n_val])
            print(f'Data split completed: {val_ratio=}')
            print(f'{n_train=}, {n_val=}')
            return train_set, val_set