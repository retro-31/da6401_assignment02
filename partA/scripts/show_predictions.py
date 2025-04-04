import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import wandb

from torchvision import transforms
from pytorch_lightning import seed_everything

# Import your model class (with the same definition you used for training)
from train import CNNClassifier  
from utils.data_loader import get_dataloaders

def show_sample_predictions(
    checkpoint_path="../models/last.ckpt",  # or "best.ckpt"
    test_dir="../data/test",
    device="cuda",
    sample_count=30  # total images to display
):
    # -----------------------------------------------------
    # 1) Load the best model checkpoint
    # -----------------------------------------------------
    model = CNNClassifier.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    # -----------------------------------------------------
    # 2) Create a test loader
    # -----------------------------------------------------
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)  # small batch for sampling

    # Class names (assuming 10 classes in iNaturalist subset)
    class_names = test_dataset.classes  

    # -----------------------------------------------------
    # 3) Gather predictions on ~sample_count images
    # -----------------------------------------------------
    rows, cols = 10, 3  # for a 10×3 grid
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 30))

    images_shown = 0
    axes = axes.flatten()  # to iterate easily over 30 subplot cells

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for i in range(images.size(0)):
                if images_shown >= sample_count:
                    break

                # Current subplot
                ax = axes[images_shown]
                img = images[i].cpu().permute(1, 2, 0).numpy()  # CHW -> HWC

                # De-normalize if you applied any normalization in transforms
                # e.g. if transforms.Normalize was used, you must invert it

                label_idx = labels[i].item()
                pred_idx = preds[i].item()

                ax.imshow(img)
                ax.set_title(f"GT: {class_names[label_idx]}\nPred: {class_names[pred_idx]}")
                ax.axis("off")

                images_shown += 1

            if images_shown >= sample_count:
                break

    fig.tight_layout()

    # -----------------------------------------------------
    # 4) Log the figure to WandB or just display it
    # -----------------------------------------------------
    # Option A: Show locally (if running in a local environment)
    # plt.show()

    # Option B: Log to WandB
    wandb.init(project="da6401-partA-sweep", name="sample_predictions")
    wandb.log({"sample_predictions_grid": wandb.Image(fig)})
    plt.close(fig)  # close the figure so it doesn't linger in memory

if __name__ == "__main__":
    # For reproducibility
    seed_everything(42)

    show_sample_predictions(
        checkpoint_path="../models/best.ckpt",  # set your best checkpoint path
        test_dir="../data/test",
        device="cuda",
        sample_count=30
    )
