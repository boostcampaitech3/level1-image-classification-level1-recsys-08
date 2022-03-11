# Image Classification

## 프로젝트 소개



### 개요

촬영된 사람 얼굴 사진에 대해 마스크 착용여부, 성별, 나이대에 따라 18개의 클래스로 구분해서 예측하기 위한 모델을 만들어 결과를 추론한다. 대회 평가 지표는 **f1 score**이다.

![56bd7d05-4eb8-4e3e-884d-18bd74dc4864..png](https://cdn.jsdelivr.net/gh/Glanceyes/ImageRepository/2022/03/11/1646962982022.png)



### 기간

2022/02/21 ~ 2022/03/03(11일)



### 팀 소개

| 팀원 | 역할 |
| --- | --- |
| 김성규 | EfficientNet 및 ensemble을 이용한 모델 |
| 이선호 | Resnet, EfficientNet, ViT를 이용한 모델 및 프로젝트 구조화 |
| 이현우 | Resnet을 이용한 모델, ArcFace loss 분석 |
| 전민규 | ResNet, Densenet을 이용한 모델 분석 |
| 정준우 | EfficientNet 및 ensemble을 이용한 모델, Label smoothing |



### 프로젝트 진행환경

- OS: Ubuntu
- GPU: V100
- DL framework: PyTorch
- IDE: Jupyter notebook, PyCharm
- 

### 데이터파일 구조

```
/input/data
├─ eval                                                    # 📁 평가 데이터
│   ├─ images
│   │   ├─ 0001b62fb1057a1182db2c839de232dbef0d1e90.jpg    # 🖼️ 총 12,600장의 이미지 파일 (40%)
│   │    ...                                               # 👥 12,600 / 7 = 1,800명의 유저
│   │   └─ fffde6a740112d7a8e81430e4a3ce06dded72993.jpg
│   └─ info.csv
└─ train                                                   # 📁 학습 데이터
    ├─ images
    │   ├─ 000001_female_Asian_45                          # 👥 2,700명의 유저(폴더)
    │   │   ├─ incorrect_mask.jpg                          # 각 유저(폴더)별 7장의 이미지 파일 존재
    │   │   ├─ mask1.jpg                                   # 🖼️ 2,700 x 7 = 총 18,900장 (60%)
    │   │   ├─ mask2.jpg                                   # 이미지 확장자는 다를 수 있습니다
    │   │   ├─ mask3.jpg
    │   │   ├─ mask4.jpg
    │   │   ├─ mask5.jpg
    │   │   └─ normal.jpg
    │   ...
    │   └─ 006959_male_Asian_19
    │        ├─ incorrect_mask.jpg
    │        ...
    │        └─ normal.jpg
    └─ train.csv
```



### 최종 제출 모델 내용 정리

- label 범위 재설정
  
    나이대 범위를 (30, 60)으로 구분한 것을 (30, 58)로 수정했다.
    
- model
  
    EfficientNet B4, Resnet152 모델로 mask, gender, age를 예측하는 multilabel classification model을 만들었다.
    
- train setting
    - loss: cross entropy loss(mask, gender), label smoothing loss(age)
    - learning rate:
    - optimizer: Adam
    - scheduler: CosineAnnealingWarmRestarts



## Repository 설명

### Repository 구조

```
code/
├─ dataset/                                  # dataset, augmentation 파일 폴더
│	  ├─ augmentation/                           # augmentation 파일 폴더
│   │   ├─ BaseAugmentation.py
│   │   ├─ CustomAugmentation.py
│   │   └─ TrainAugmentation.py
│   ├─ BaseDataset.py
│   ├─ SplitByProfileDataset.py
│   └─ TestDataset.py
├─ inference/                                # inference 파일 폴더
│	  └─ Inferrer.py
├─ loss/                                     # loss 파일 폴더
│	  └─ loss.py
├─ models/                                   # model 파일 폴더
│	  ├─ BaseModel.py
│	  ├─ EfficientNetB3.py
│	  ├─ EfficientNetB4.py
│	  ├─ ResNet18.py
│	  ├─ ResNet50.py
│	  └─ VisionTransformer.py
├─ schedulers/                               # scheduler 파일 폴더
│	  ├─ CosineAnnealing.py        
│	  ├─ CosineAnnealingWarmRestarts.py        
│	  └─ StepLR.py        
├─ train/                                    # train 파일 폴더
│	  ├─ Trainer.py  
│	  └─ Validator.py   
├─ utils/                                    # util 파일 폴더
│	  ├─ setConfig.py  
│	  └─ util.py   
├─ config.ini     
└─ run.py
```



### **Dependency**

- torch==1.10.2
- torchvision==0.11.1
- timm==****0.5.4****



### **Install Requirements**

```bash
pip install -r requirements.txt
```



### 실행 방법

1. `code/config.ini` 파일에서 파라미터 설정을 한다.
2. 아래의 코드를 실행한다.
   
    ```bash
    python code/run.py
    ```