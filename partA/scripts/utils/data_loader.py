from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir, batch_size=32, val_split=0.2, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
