import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def grid_image(np_images, gts, preds, dataset, n=16, shuffle=False):
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
        gt_decoded_labels = dataset.decode_multi_class(gt)
        pred_decoded_labels = dataset.decode_multi_class(pred)
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


class Validator:
    def __init__(self, args, device):
        self.args = args
        self.device = device

    def validate(self, model, criterion, dataset, val_dataset, val_loader):
        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            total_labels = []
            total_preds = []

            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                total_labels.append(labels.cpu())
                total_preds.append(preds.cpu())

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()

                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, dataset, n=16,
                        shuffle=True
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_dataset)

            total_labels = np.concatenate(total_labels)
            total_preds = np.concatenate(total_preds)
            val_f1 = f1_score(total_labels, total_preds, average='macro')

        return val_loss, val_acc, val_f1, figure
