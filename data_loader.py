import torch
import torchvision
import torchvision.transforms as transforms

def load_data(batch_size):
    # Define data transformation
    transform = transforms.ToTensor()

    # Load data set
    train_data = torchvision.datasets.MNIST(root="./data", train=True,  transform=transform, download=True)
    test_data = torchvision.datasets.MNIST(root="./data", train=False,  transform=transform, download=True)

    # Load data using torch
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader, test_loader