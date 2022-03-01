import torch
import numpy as np
from tqdm import tqdm
from .Validator import Validator


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Trainer:
    def __init__(self, args, model, criterion, optimizer,
                 train_loader, valid_loader, device, logger, lr_scheduler=None):

        self.args = args
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.logger = logger
        self.lr_scheduler = lr_scheduler

        self.validator = Validator(args, device)

    def train(self, dataset, val_dataset):
        print("Training Started!")
        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(self.args.epochs):
            print(f"epoch: {epoch}")
            # train loop
            self.model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(tqdm(self.train_loader)):
                print(f"idx: {idx}")
                inputs, labels = train_batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outs = self.model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = self.criterion(outs, labels)

                loss.backward()
                self.optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % self.args.log_interval == 0:
                    train_loss = loss_value / self.args.log_interval
                    train_acc = matches / self.args.train_batch_size / self.args.log_interval
                    current_lr = get_lr(self.optimizer)
                    print(
                        f"Epoch[{epoch}/{self.args.epochs}]({idx + 1}/{len(self.train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    self.logger.add_scalar("Train/loss", train_loss, epoch * len(self.train_loader) + idx)
                    self.logger.add_scalar("Train/accuracy", train_acc, epoch * len(self.train_loader) + idx)

                    loss_value = 0
                    matches = 0

            self.lr_scheduler.step()

            val_loss, val_acc, figure = self.validator.validate(model=self.model, criterion=self.criterion,
                                                                dataset=dataset, val_dataset=val_dataset,
                                                                val_loader=self.valid_loader
                                                                )

            best_val_loss = min(best_val_loss, val_loss)

            if val_acc > best_val_acc:
                print(f"New best models for val accuracy : {val_acc:4.2%}! saving the best models..")
                torch.save(self.model.module.state_dict(), f"{self.args.result_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(self.model.module.state_dict(), f"{self.args.result_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            self.logger.add_scalar("Val/loss", val_loss, epoch)
            self.logger.add_scalar("Val/accuracy", val_acc, epoch)
            self.logger.add_figure("results", figure, epoch)
            print()
