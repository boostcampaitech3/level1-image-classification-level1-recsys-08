import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..dataset import TestDataset, MaskBaseDataset


class Inference:
    def __init__(self, test_data_dir, test_data_file, model, output_dir, device, args):
        self.args = args
        self.test_data_dir = test_data_dir
        self.test_data_file = test_data_file
        self.model = model
        self.output_dir = output_dir
        self.device = device
        self.info = pd.read_csv(test_data_file)
        self.model.to(self.device)

    @torch.no_grad()
    def inference(self, test_loader):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        self.model.eval()

        print("Calculating inference results..")
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(test_loader):
                images = images.to(device)
                pred = self.model(images)
                pred = pred.argmax(dim=-1)
                preds.extend(pred.cpu().numpy())

        self.info['ans'] = preds
        save_path = os.path.join(output_dir, f'output.csv')
        self.info.to_csv(save_path, index=False)
        print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and models checkpoints directories
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for validing (default: 128)')
    parser.add_argument('--resize', type=tuple, default=(128, 96),
                        help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--models', type=str, default='BaseModel',
                        help='models type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './experiments/results'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
