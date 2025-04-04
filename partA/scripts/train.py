import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
import torch.nn.functional as F
import wandb
import argparse
from utils.model import FlexibleCNN
from utils.data_loader import get_dataloaders
import torch

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
        loss = F.cross_entropy(preds, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        acc = (preds.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    wandb.init()

    # WandB config for hyperparameters
    config = wandb.config

    wandb_logger = WandbLogger(project="da6401-partA-sweep")
    checkpoint_callback = ModelCheckpoint(dirpath='../models', save_top_k=1, monitor='val_acc', mode='max')

    train_loader, val_loader = get_dataloaders("../data/train", batch_size=config.batch_size)

    model_params = {
        'num_conv_layers': config.num_conv_layers,
        'num_filters': config.num_filters,
        'kernel_size': config.kernel_size,
        'activation': getattr(nn, config.activation),
        'dense_neurons': config.dense_neurons,
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

if __name__ == "__main__":
    main()
