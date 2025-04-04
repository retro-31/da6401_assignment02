import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_finetune import FineTuneCNN
import wandb

def show_predictions(checkpoint='../models/best.ckpt', test_dir='../data/test', device='cuda'):
    model = FineTuneCNN.load_from_checkpoint(checkpoint).to(device)
    model.eval()

    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    loader = DataLoader(test_dataset, batch_size=30, shuffle=True)
    images, labels = next(iter(loader))
    preds = model(images.to(device)).argmax(dim=1).cpu()

    class_names = test_dataset.classes

    fig, axes = plt.subplots(10,3, figsize=(12,30))
    axes = axes.flatten()
    for i in range(30):
        img = images[i].permute(1,2,0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"GT: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")

    plt.tight_layout()
    wandb.init(project="da6401-partB-sweep", name="partB_predictions")
    wandb.log({"sample_predictions": wandb.Image(fig)})
    plt.close()

if __name__ == "__main__":
    show_predictions()