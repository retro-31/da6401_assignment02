import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.model import FlexibleCNN
from utils.data_loader import get_dataloaders
import torch.nn as nn
import torch
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import shutil
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class CNNClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3, **model_params):
        super().__init__()
        self.model = FlexibleCNN(**model_params)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.functional.cross_entropy(preds, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.functional.cross_entropy(preds, y)
        acc = (preds.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def evaluate_test(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    return test_acc

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_filters", type=int, default=32)
    parser.add_argument("--activation", type=str, default="ReLU")
    parser.add_argument("--filter_organisation", type=str, default="same")
    parser.add_argument("--data_augmentation", type=str2bool, default=False,
                        help="Use data augmentation (True/False)")
    parser.add_argument("--use_batchnorm", type=str2bool, default=False,
                        help="Use BatchNorm (True/False)")
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)

    args = parser.parse_args()

    run_name = (
        f"filters_{args.num_filters}_"
        f"act_{args.activation}_"
        f"{args.filter_organisation}_"
        f"aug_{args.data_augmentation}_"
        f"bn_{args.use_batchnorm}_"
        f"drop_{args.dropout_rate}_"
        f"bs_{args.batch_size}_"
        f"lr_{args.lr:.0e}"
    )

    wandb.init(project="da6401-partA-sweep", config=vars(args), name=run_name)

    config = wandb.config
    wandb_logger = WandbLogger(project="da6401-partA-sweep")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='../models',
        filename='best_{val_acc:.3f}',
        save_top_k=1,
        monitor='val_acc',
        mode='max'
    )

    activation_cls = getattr(nn, config.activation)

    train_loader, val_loader = get_dataloaders(
        "../data/train",
        batch_size=config.batch_size,
        data_augmentation=config.data_augmentation
    )

    model_params = {
        'num_filters': config.num_filters,
        'activation': activation_cls,
        'filter_organisation': config.filter_organisation,
        'use_batchnorm': config.use_batchnorm,
        'dropout_rate': config.dropout_rate,
        'num_classes': 10
    }

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    model = CNNClassifier(lr=config.lr, **model_params)
    trainer.fit(model, train_loader, val_loader)

    # Evaluate on test data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_dataset = datasets.ImageFolder("../data/test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_accuracy = evaluate_test(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    wandb.log({"test_acc": test_accuracy})

    # Explicitly copy best checkpoint as "best.ckpt"
    best_model_path = checkpoint_callback.best_model_path
    final_best_path = '../models/best.ckpt'

    if best_model_path:
        shutil.copy(best_model_path, final_best_path)
        print(f"Best model saved explicitly as {final_best_path}")

if __name__ == "__main__":
    main()
