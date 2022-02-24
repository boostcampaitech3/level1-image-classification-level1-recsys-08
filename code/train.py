import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim,nn
from torchvision import models,transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskDataset, AddGaussianNoise
from models import Pretrained_Model


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
def train(data_dir, model_dir, args):
    
    seed_everything(args.seed)
    save_dir = os.path.join('../model', args.name)
    
    # cuda 사용 여부 setting
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # dataset 불러오기 / train_loader 정의
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([transforms.RandomRotation(degrees = 15),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_std),
                                          AddGaussianNoise()])
    dataset = MaskDataset(data_dir=data_dir,transforms=train_transform, adj_csv = True, val_ratio = args.val_ratio, up_sampling = 3)
    train_set, val_set = dataset.split_dataset()
    num_classes = 18
    batch_size = args.batch_size
    train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
                            )
    val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
)
    # model 불러오기
    model = Pretrained_Model(args.model_name, num_classes).to(device)
    model = torch.nn.DataParallel(model) # 무슨 역할??????
    
    # loss & metric & scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)
#     scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    print("Training starts...")
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        scheduler.step()
        
        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
#             figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

#                 if figure is None:
#                     inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
#                     inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
#                     figure = grid_image(
#                         inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
#                     )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
#             logger.add_figure("results", figure, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    print('hello')
    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=12, help='number of epochs to train (default: 12)')
#         parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
#         parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
#         parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
#         parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model_name', type=str, default='resnet50', help='model type (default: Resnet50)')
#         parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
#         parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
#         parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
#         parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/train')
    parser.add_argument('--model_dir', type=str, default='../model')

    # Container environment
#         parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
#         parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)