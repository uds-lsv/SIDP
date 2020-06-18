from torchvision import datasets
from torch.utils.data import DataLoader, random_split, RandomSampler
import torchvision.transforms as transforms


def data_loaders(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    sampler = RandomSampler(train_data, replacement=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8, sampler=sampler, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=1)

    return train_loader, test_loader

def axi_loader(batch_size):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    axi_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    axi_loader = DataLoader(axi_data, batch_size=batch_size)
    axi_x, _ = next(iter(axi_loader))
    return  axi_x


