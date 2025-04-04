from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir, batch_size=32, val_split=0.2, image_size=224, data_augmentation=True, num_workers=4):
    transform_list = [transforms.Resize((image_size, image_size))]
    
    if data_augmentation:
        transform_list += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)
        ]

    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
