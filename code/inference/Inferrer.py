import os
import pandas as pd
import torch


class Inferrer:
    def __init__(self, test_data_dir, test_data_file, model, output_dir, device, args):
        self.args = args
        self.test_data_dir = test_data_dir
        self.test_data_file = test_data_file
        self.model = model
        self.output_dir = output_dir
        self.device = device
        self.info = pd.read_csv(test_data_file)
        self.model.to(self.device)

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
        save_path = os.path.join(self.output_dir, f'output_{self.args.name}.csv')
        self.info.to_csv(save_path, index=False)
        print(f"Inference Done! Inference result saved at {save_path}")
