import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from utils.data_loader import get_dataloaders
import wandb
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import shutil

class FineTuneCNN(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-4, unfreeze_layers=0):
        super().__init__()
        self.lr = lr
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        layers = list(self.model.children())
        if unfreeze_layers > 0:
            # Unfreeze the last 'unfreeze_layers' layers
            for layer in layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # Freeze the remaining layers
            for layer in layers[:-unfreeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the classifier to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Only update parameters that require gradients
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

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
    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unfreeze_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    # Construct a meaningful run name for WandB
    run_name = f"unfreeze_{args.unfreeze_layers}_bs_{args.batch_size}_lr_{args.lr:.0e}"
    wandb.init(project="da6401-partB-sweep", config=vars(args), name=run_name)
    wandb_logger = WandbLogger(project="da6401-partB-sweep")

    # Set up a checkpoint callback that saves the best model (based on validation accuracy)
    checkpoint_callback = ModelCheckpoint(
        dirpath='../models', 
        save_top_k=1, 
        monitor='val_acc', 
        mode='max', 
        filename='best'
    )

    train_loader, val_loader = get_dataloaders("../data/train", batch_size=args.batch_size)

    model = FineTuneCNN(lr=args.lr, unfreeze_layers=args.unfreeze_layers)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs, 
        accelerator='gpu', 
        devices=1,
        precision='16-mixed', 
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_loader, val_loader)

    # Evaluate on test data after training
    test_transform = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor()
    ])
    test_dataset = datasets.ImageFolder("../data/test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_acc = evaluate_test(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    wandb.log({"test_acc": test_acc})

    # Copy the best checkpoint explicitly as "best.ckpt" for use in prediction scripts
    if checkpoint_callback.best_model_path:
        shutil.copy(checkpoint_callback.best_model_path, "../models/best.ckpt")
        print(f"Best model saved explicitly as ../models/best.ckpt")

if __name__ == "__main__":
    main()
