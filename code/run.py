import argparse
import ast
import configparser
import multiprocessing
import os
from importlib import import_module

from dotenv import load_dotenv
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from loss.loss import create_criterion
from train.Trainer import Trainer
from inference.Inferrer import Inferrer
# utils
from utils.util import *
from utils.setConfig import *


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module(f"models.{args.model}"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def train(train_data_dir: str, result_dir:str, args):
    seed_everything(args.seed)

    result_dir = increment_path(os.path.join(result_dir, args.name))
    args.result_dir = result_dir

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
    transform_module = getattr(import_module(f"dataset.augmentation.{args.train_augmentation}"),
                               args.train_augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
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
    )

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


def infer(test_data_dir: str, test_data_file: str, model_dir: str, output_dir: str, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = args.num_classes  # 18
    model = load_model(model_dir, num_classes, device)

    test_dataset_module = getattr(import_module(f"dataset.{args.test_dataset}"), args.test_dataset)
    test_dataset = test_dataset_module(
        test_data_dir=test_data_dir,
        test_data_file=test_data_file
    )

    # -- augmentation
    transform_module = getattr(import_module(f"dataset.augmentation.{args.test_augmentation}"),
                               args.test_augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    test_dataset.set_transform(transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    inferrer = Inferrer(test_data_dir, test_data_file, model, output_dir, device, args)
    inferrer.inference(test_loader)


if __name__ == '__main__':
    load_dotenv(verbose=True)
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--train_dataset', type=str, default='BaseDataset',
                        help='dataset type (default: BaseDataset)')
    parser.add_argument('--train_data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/MaskClassification/data/train/images'))
    parser.add_argument('--train_data_file', type=str, default=None,
                        help='set directory of ".csv" file of train dataset if it exists')
    parser.add_argument('--test_dataset', type=str, default='TestDataset',
                        help='test dataset type (default: TestDataset)')
    parser.add_argument('--test_data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/MaskClassification/data/eval/images'))
    parser.add_argument('--test_data_file', type=str, default='info.csv',
                        help='set directory of ".csv" file of test dataset if it exists')
    parser.add_argument('--train_augmentation', type=str, default='BaseAugmentation',
                        help='train data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--test_augmentation', type=str, default='BaseAugmentation',
                        help='test data augmentation type (default: BaseAugmentation)')
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

    # test
    parser.add_argument('--test_batch_size', type=int, default=128,
                        help='input batch size for inference (default: 128)')
    parser.add_argument('--output_dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/MaskClassification/output'))

    # model
    parser.add_argument('--model', type=str, default='BaseModel',
                        help='models type (default: BaseModel)')
    parser.add_argument('--model_parameter', type=json.loads, default=None,
                        help='set parameters of model network')

    # Check config.ini file
    parser.add_argument('--config_file', type=str,
                        default=os.environ.get('SM_CONFIG_FILE', '/opt/ml/MaskClassification/code/config.ini'))

    # project
    parser.add_argument('--name', default='BaseModel',
                        help='models save at {result_dir}/{name}')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')

    args = parser.parse_args()

    if args.config_file:
        config_namespace = argparse.Namespace()
        with open(args.config_file, 'r') as config_file:
            config = configparser.ConfigParser()
            config.read(args.config_file)

            if 'project' in config.sections():
                # string
                set_config_as_string(config_namespace, config, 'project', 'name')

                # integer
                set_config_as_int(config_namespace, config, 'project', 'seed')

            if 'data' in config.sections():
                # string
                set_config_as_string(config_namespace, config, 'data', 'train_dataset')
                set_config_as_string(config_namespace, config, 'data', 'train_data_dir')
                set_config_as_string(config_namespace, config, 'data', 'train_data_file')
                set_config_as_string(config_namespace, config, 'data', 'test_dataset')
                set_config_as_string(config_namespace, config, 'data', 'test_data_dir')
                set_config_as_string(config_namespace, config, 'data', 'test_data_file')
                set_config_as_string(config_namespace, config, 'data', 'train_augmentation')
                set_config_as_string(config_namespace, config, 'data', 'test_augmentation')

                # integer
                set_config_as_int(config_namespace, config, 'data', 'num_classes')

                # json
                set_config_as_json(config_namespace, config, 'data', 'resize')

            if 'train' in config.sections():
                # float
                set_config_as_float(config_namespace, config, 'train', 'lr')

                # string
                set_config_as_string(config_namespace, config, 'train', 'criterion')
                set_config_as_string(config_namespace, config, 'train', 'optimizer')
                set_config_as_string(config_namespace, config, 'train', 'result_dir')

                # integer
                set_config_as_int(config_namespace, config, 'train', 'epochs')
                set_config_as_int(config_namespace, config, 'train', 'train_batch_size')

            if 'valid' in config.sections():
                # integer
                set_config_as_int(config_namespace, config, 'valid', 'valid_batch_size')

            if 'test' in config.sections():
                # string
                set_config_as_string(config_namespace, config, 'test', 'output_dir')

                # integer
                set_config_as_int(config_namespace, config, 'test', 'test_bacth_size')

            if 'model' in config.sections():
                # string
                set_config_as_string(config_namespace, config, 'model', 'model')

                # boolean
                set_config_as_bool(config_namespace, config, 'model', 'pre_trained')

                model_parameter = dict()
                for key, value in config.items('model'):
                    if key not in ['model', 'pre_trained']:
                        value = ast.literal_eval(value)
                        model_parameter[key] = value
                setattr(config_namespace, 'model_parameter', model_parameter)

        args = parser.parse_args(namespace=config_namespace)

    print(args)

    train_data_dir = args.train_data_dir

    test_data_dir = args.test_data_dir
    test_data_file = args.test_data_file
    output_dir = args.output_dir
    result_dir = args.result_dir

    torch.cuda.empty_cache()
    train(train_data_dir, result_dir, args)

    result_dir = args.result_dir
    torch.cuda.empty_cache()
    infer(test_data_dir, test_data_file, result_dir, output_dir, args)