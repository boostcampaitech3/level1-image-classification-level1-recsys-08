# **팀명정해조** 마스크 착용 상태 분류 프로젝트

## 프로젝트 소개



### 프로젝트 개요
- 촬영된 사람 얼굴 사진에 대해 마스크 착용여부, 성별, 나이대에 따라 18개의 클래스로 구분하여 예측하기 위한 모델을 만들어 결과를 추론한다.
- f1 score로 모델 성능을 평가한다.



### 팀 목표

- 모델을 제작할 때 사용되는 데이터의 특징이 무엇이고 어떠한 인사이트를 얻을 수 있는지를 EDA를 통해 해석한다. 
- 팀원 개별적으로 대회에서 주어진 Label을 가장 잘 분류할 수 있는 최적의 모델을 찾고, 이를 바탕으로 다양한 Hyperparameter와 학습 기법을 적용하여 모델의 정확도와 F1 Score을 높이는 것을 목표로 한다.



### 예측 클래스 설명
![class](https://cdn.jsdelivr.net/gh/Glanceyes/ImageRepository/2022/03/11/1646962982022.png)



### 프로젝트 기간

2022.02.21 ~ 2022.03.03(11일)



### 프로젝트 진행환경

- OS: Ubuntu
- GPU: V100
- DL framework: PyTorch
- IDE: Jupyter notebook, PyCharm



### 프로젝트 요약

- **전처리**

    나이대 범위를 (0 ~ 29, 30 ~ 59, 60이상)에서 (0 ~ 29, 30 ~ 57, 58이상)로 수정했다.
    
- **모델**

    EfficientNet B4, ResNet 152를 사용했다.

    마스크 착용여부, 성별, 나이대를 예측할 수 있는 multilabel classification model 2개를 Ensemble하였다.
    
- **Hyperparameter**
  
    - loss: cross entropy loss(마스크 착용예측, 성별), label smoothing loss(나이대)
    - optimizer: Adam
    - scheduler: CosineAnnealingWarmRestarts



### 프로젝트 결과

| rank | f1 score | accuracy |
|:----:|:--------:|:--------:|
|  16  |  0.7428  |  79.9683 | 



### 데이터 구조

```
/input/data
├─ eval                                                    # 📁 평가 데이터
│   ├─ images
│   │   ├─ 0001b62fb1057a1182db2c839de232dbef0d1e90.jpg    # 🖼️ 총 12,600장의 이미지 파일 (40%)
│   │    ...                                               # 👥 12,600 / 7 = 1,800명의 유저
│   │   └─ fffde6a740112d7a8e81430e4a3ce06dded72993.jpg
│   └─ info.csv
└─ train                                                   # 📁 학습 데이터
    ├─ images
    │   ├─ 000001_female_Asian_45                          # 👥 2,700명의 유저(폴더)
    │   │   ├─ incorrect_mask.jpg                          # 각 유저(폴더)별 7장의 이미지 파일 존재
    │   │   ├─ mask1.jpg                                   # 🖼️ 2,700 x 7 = 총 18,900장 (60%)
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



### Flow Chart

![mermaid-diagram-20220311110915](https://cdn.jsdelivr.net/gh/Glanceyes/ImageRepository/2022/03/11/1646964624584.png)



## 팀 소개

| [ ![김성규](https://avatars.githubusercontent.com/u/69254522?v=4) ](https://github.com/hikible) | [ ![이선호](https://avatars.githubusercontent.com/u/65075134?v=4) ](https://github.com/Glanceyes) | [ ![이현우](https://avatars.githubusercontent.com/u/52898220?v=4) ](https://github.com/harrier999) | [ ![전민규](https://avatars.githubusercontent.com/u/85151359?v=4) ](https://github.com/alsrb0607) | [ ![정준우](https://avatars.githubusercontent.com/u/39089969?v=4) ](https://github.com/ler0n) |
|:-----------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|
|                              [ 김성규 ](https://github.com/hikible)                             |                              [ 이선호 ](https://github.com/Glanceyes)                             |                              [ 이현우 ](https://github.com/harrier999)                             |                              [ 전민규 ](https://github.com/alsrb0607)                             |                              [ 정준우 ](https://github.com/ler0n)                             |
|                              EfficientNet, ResNet 모델 분석 및 ensemble 진행                             |                     ResNet, EfficientNet, ViT를 이용한 모델 분석 및 프로젝트 구조화                    |                               ResNet을 이용한 모델, ArcFace loss 분석                              |                                ResNet, DenseNet을 이용한 모델 분석                                |                    모델 ensemble, Label smoothing loss 커스텀                    |



## Repository 설명

### 구조
```
code/
├─ dataset/             
│   ├─ augmentation/
│   │   ├─ BaseAugmentation.py
│   │   ├─ CustomAugmentation.py
│   │   └─ TrainAugmentation.py
│   ├─ BaseDataset.py
│   ├─ SplitByProfileDataset.py
│   └─ TestDataset.py
├─ inference/        
│   └─ Inferrer.py
├─ loss/    
│   └─ loss.py
├─ models/    
│   ├─ BaseModel.py
│   ├─ EfficientNetB3.py
│   ├─ EfficientNetB4.py
│   ├─ EfficientNetB4T.py
│   ├─ ResNet152.py
│   ├─ ResNet18.py
│   ├─ ResNet50.py
│   └─ VisionTransformer.py
├─ schedulers/     
│   ├─ CosineAnnealing.py        
│   ├─ CosineAnnealingWarmRestarts.py        
│   └─ StepLR.py        
├─ train/    
│   ├─ Trainer.py  
│   └─ Validator.py   
├─ utils/   
│   ├─ setConfig.py  
│   └─ util.py   
├─ config.ini     
└─ run.py
```



### Dependencies

- torch==1.10.2 
- torchvision==0.11.1 
- tensorboard==2.4.1 
- pandas==1.1.5 
- opencv-python==4.5.1.48 
- scikit-learn~=0.24.1 
- matplotlib==3.2.1 
- numpy~=1.18.0 
- einops~=0.4.0 
- Pillow~=8.1.0 
- python-dotenv~=0.19.2 
- tqdm~=4.51.0 
- timm==0.5.4 



### 실행 방법

1. requirement 설치하기
    ```bash
    pip install -r requirements.txt
    ```
2. `code/config.ini` 파일로 모델 학습 설정하기
3. `run.py` 실행하기
    ```bash
    python code/run.py
    ```