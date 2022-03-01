import argparse
import ast
import configparser
import json
import multiprocessing
import os
from importlib import import_module

from dotenv import load_dotenv
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from loss.loss import create_criterion
from train.Trainer import Trainer
# utils
from utils.util import *

load_dotenv(verbose=True)


def train(train_data_dir, result_dir, args):
    seed_everything(args.seed)

    result_dir = increment_path(os.path.join(result_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module(f"dataset.{args.train_dataset}"), args.train_dataset)  # default: BaseDataset
    dataset = dataset_module(
        data_dir=train_data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module(f"dataset.augmentation.{args.augmentation}"),
                               args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_dataset = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    valid_loader = DataLoader(
        val_dataset,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- models
    model_module = getattr(import_module(f"models.{args.model}"), args.model)  # default: BaseModel
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
    lr_scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=result_dir)
    with open(os.path.join(result_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    trainer = Trainer(args, model, criterion, optimizer, train_loader, valid_loader, device, logger, lr_scheduler)
    trainer.train(dataset, val_dataset)


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
