from torchvision import datasets
from torch.utils.data import DataLoader, random_split, RandomSampler
import torchvision.transforms as transforms


def data_loaders(batch_size):
    transform = transforms.Compose([transforms.ToTensor(),])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    sampler = RandomSampler(train_data, replacement=True)
    train_loader = DataLoader(train_data, batch_size, num_workers=8, sampler=sampler)
    test_loader = DataLoader(test_data, batch_size=1)

    return train_loader, test_loader

def axi_loader(batch_size):
    transform = transforms.Compose([transforms.ToTensor(),])
    axi_data = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    axi_loader = DataLoader(axi_data, batch_size=batch_size)
    axi_x, _ = next(iter(axi_loader))

    return  axi_x


