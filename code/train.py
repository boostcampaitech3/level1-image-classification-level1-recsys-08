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

from dataset import MaskDataset, AddGaussianNoise, MaskDatasetForSplit
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
    
def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"
    
    
def train(data_dir, model_dir, args):
    
    seed_everything(args.seed)
    save_dir = increment_path(os.path.join('../model', args.name))
    
    # cuda 사용 여부 setting
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # dataset 불러오기 / train_loader 정의
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_std),
#                                           AddGaussianNoise()
    ])
    batch_size = args.batch_size
    # model 불러오기
    if not args.split_model:
        model = Pretrained_Model(args.model_name, 18)
#         model = torch.nn.DataParallel(model) # 무슨 역할??????
        models = [model]
    else:
        model_mask = Pretrained_Model(args.model_name, 3)
        model_gender = Pretrained_Model(args.model_name, 2)
        model_age = Pretrained_Model(args.model_name, 3)
        models = [model_mask, model_gender, model_age]
    models_info = ['Mask', 'Gender', 'Age']
    # loss & metric & scheduler
    
    

    # logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    for i, model in enumerate(models):
        best_val_acc = 0
        best_val_loss = np.inf
        model_now = models_info[i]
        model = models[i].to(device)
        model = torch.nn.DataParallel(model)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss().to(device)
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
        dataset = MaskDatasetForSplit(data_dir=data_dir,transforms=train_transform, adj_csv = True, val_ratio = args.val_ratio, up_sampling = args.up_sampling, key = model_now)
        train_set, val_set = dataset.split_dataset()
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
        print(f"Training for {model_now} starts...")
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
#             if epoch+1%6==0:
#                 for name, param in model.named_parameters():
#                     param.requires_grad = True
#             else:
            for name, param in model.named_parameters():
                if 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
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
                    torch.save(model.module.state_dict(), f"{save_dir}/best_{model_now}.pth")
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{save_dir}/last_{model_now}.pth")
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
    parser.add_argument('--up_sampling', type=int, default=0, help='up_sample age_label 1 & 2 by up_sampling times (default: 0)')
    parser.add_argument('--epochs', type=int, default=4, help='number of epochs to train (default: 4)')
#         parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
#         parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
#         parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
#         parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model_name', type=str, default='resnet18', help='model type (default: Resnet18)')
    parser.add_argument('--split_model', type=bool, default='False', help='split model into 3 parts (default: False)')
#         parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
#         parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
#         parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
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