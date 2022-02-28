import argparse
import ast
import configparser
import json
import multiprocessing
import os
from importlib import import_module

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.BaseDataset import BaseDataset
from loss.loss import create_criterion
# utils
from utils.util import *

load_dotenv(verbose=True)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize를 조정해야 할 수 있습니다.
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top을 조정해야 할 수 있습니다.
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]

    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = BaseDataset.decode_multi_class(gt)
        pred_decoded_labels = BaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def train(train_data_dir, result_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(result_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module(f"dataset.{args.train_dataset}"), args.train_dataset)  # default: BaseDataset
    _dataset = dataset_module(
        data_dir=train_data_dir,
    )
    num_classes = _dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module(f"dataset.augmentation.{args.augmentation}"),
                               args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=_dataset.mean,
        std=_dataset.std,
    )
    _dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = _dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
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

    # -- models
    model_module = getattr(import_module(f"models.{args.model}"), args.model)  # default: BaseModel
    model = None
    model = model_module(
        num_classes=num_classes,
        **args.model_parameter
    ).to(device)

    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
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
                train_acc = matches / args.train_batch_size / args.log_interval
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
            figure = None
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

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, _dataset.mean, _dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.train_dataset != "SplitByProfileDataset"
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best models for val accuracy : {val_acc:4.2%}! saving the best models..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and models checkpoints directories

    # project
    parser.add_argument('--name', default='BaseModel',
                        help='models save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')

    # data
    parser.add_argument('--train_dataset', type=str, default='BaseDataset',
                        help='dataset type (default: BaseDataset)')
    parser.add_argument('--train_data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/MaskClassification/data/train/images'))
    parser.add_argument('--train_data_csv', type=str, default=None,
                        help='set directory of ".csv" file of train dataset if it exists')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation',
                        help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96],
                        help='resize size for image when training')

    # train
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--criterion', type=str, default='cross_entropy',
                        help='criterion type (default: cross_entropy)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer type (default: Adam)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr_decay_step', type=int, default=20,
                        help='learning rate scheduler decay step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--result_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/MaskClassification/experiments/results'))

    # valid
    parser.add_argument('--valid_batch_size', type=int, default=128,
                        help='input batch size for validing (default: 128)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='ratio for validaton (default: 0.2)')

    # model
    parser.add_argument('--model', type=str, default='BaseModel',
                        help='models type (default: BaseModel)')
    parser.add_argument('--model_parameter', type=json.loads, default=None,
                        help='set parameters of model network')

    # Check config.ini file
    parser.add_argument('--config_file', type=str,
                        default=os.environ.get('SM_CONFIG_FILE', '/opt/ml/MaskClassification/code/config.ini'))

    args = parser.parse_args()



    if args.config_file:
        config_namespace = argparse.Namespace()
        with open(args.config_file, 'r') as config_file:
            config = configparser.ConfigParser()
            config.read(args.config_file)

            if 'project' in config.sections():
                # string
                setattr(config_namespace, 'name', config.get('project', 'name'))

                # integer
                setattr(config_namespace, 'seed', config.getint('project', 'seed'))

            if 'data' in config.sections():
                # string
                setattr(config_namespace, 'train_dataset', config.get('data', 'train_dataset'))
                setattr(config_namespace, 'train_data_dir', config.get('data', 'train_data_dir'))
                setattr(config_namespace, 'train_data_csv', config.get('data', 'train_data_csv'))
                setattr(config_namespace, 'augmentation', config.get('data', 'augmentation'))

                # integer
                setattr(config_namespace, 'num_classes', config.getint('data', 'num_classes'))

            if 'train' in config.sections():
                # float
                setattr(config_namespace, 'lr', config.getfloat('train', 'lr'))

                # string
                setattr(config_namespace, 'criterion', config.get('train', 'criterion'))
                setattr(config_namespace, 'optimizer', config.get('train', 'optimizer'))
                setattr(config_namespace, 'result_dir', config.get('train', 'result_dir'))

                # integer
                setattr(config_namespace, 'epochs', config.getint('train', 'epochs'))
                setattr(config_namespace, 'train_batch_size', config.getint('train', 'train_batch_size'))

            if 'valid' in config.sections():
                # integer
                setattr(config_namespace, 'valid_batch_size', config.getint('valid', 'valid_batch_size'))

            if 'model' in config.sections():
                # string
                setattr(config_namespace, 'model', config.get('model', 'model'))

                # boolean
                setattr(config_namespace, 'pre_trained', config.getboolean('model', 'pre_trained'))

                model_parameter = dict()
                for key, value in config.items('model'):
                    if key not in ['model', 'pre_trained']:
                        value = ast.literal_eval(value)
                        model_parameter[key] = value
                setattr(config_namespace, 'model_parameter', model_parameter)

        args = parser.parse_args(namespace=config_namespace)

    print(args)

    train_data_dir = args.train_data_dir
    result_dir = args.result_dir

    train(train_data_dir, result_dir, args)
