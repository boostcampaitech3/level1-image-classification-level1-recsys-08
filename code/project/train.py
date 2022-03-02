import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from augmentation import TrainAugmentation, EvalAugmentation
from dataset import MaskBaseDataset
from loss import create_criterion
from sampler import ImbalancedSampler


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

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
        label_mode=args.label_mode
    )
    num_classes = 3 if args.label_mode != 1 else 2
    print(num_classes)
    
    # -- augmentation
    transform_module = getattr(import_module("augmentation"), args.augmentation)  # default: BaseAugmentation
    train_transform = transform_module()
    val_transform = EvalAugmentation()

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        sampler=ImbalancedSampler(train_set),
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion, classes=num_classes)
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_f1 = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        f1_value = 0
        
        train_set.dataset.set_transform(train_transform)
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
            f1_value += f1_score(labels.cpu(), preds.cpu(), average='macro')
            
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_f1 = f1_value / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training f1 score {train_f1:4.4} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/f1 score", train_f1, epoch * len(train_loader) + idx)

                loss_value = 0
                f1_value = 0

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            val_set.dataset.set_transform(val_transform)
            model.eval()
            val_loss_items = []
            val_f1_items = []
            # figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                f1_item = f1_score(labels.cpu(), preds.cpu(), average='macro')
                val_loss_items.append(loss_item)
                val_f1_items.append(f1_item)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_f1 = np.sum(val_f1_items) / len(val_loader)
            best_val_loss = min(best_val_loss, val_loss)
            if val_f1 > best_val_f1:
                print(f"New best model for val f1 score : {val_f1:4.4}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_f1 = val_f1
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] f1 score : {val_f1:4.4}, loss: {val_loss:4.2} || "
                f"best f1 score : {best_val_f1:4.4}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/f1_score", val_f1, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset',
                        help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='TrainAugmentation',
                        help='train data augmentation type (default: TrainAugmentation)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64,
                        help='input batch size for validing (default: 64)')
    parser.add_argument('--model', type=str, default='EfficientModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=0.00006, help='learning rate (default: 0.00006)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='ce-f1',
                        help='criterion type')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--label_mode', default=2, type=int, help='label mode(0=mask, 1=gender, 2=age')
    parser.add_argument('--name', default='model', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
