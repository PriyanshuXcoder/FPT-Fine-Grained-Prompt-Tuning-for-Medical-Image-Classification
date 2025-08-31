import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def get_data_loaders(data_dir, batch_size=32, input_size=(224, 224)):
    high_res_transform = transforms.Compose([
        transforms.Resize(input_size), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=high_res_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=high_res_transform)
    test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=high_res_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
