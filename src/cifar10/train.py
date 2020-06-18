import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, RandomSampler
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from .dataset import data_loaders, axi_loader


def train(model, criterion, optimizer, device, train_loader, clip, scheduler_lr,
          noise_multiplier, batch_size, axi_x):
    model.train()
    for x, y in train_loader:
        _lr = optimizer.param_groups[0]['lr']
        std_params = _lr * clip * noise_multiplier / batch_size
        x = torch.cat([x.to(device), axi_x], dim=0)
        y = y.to(device)

        optimizer.zero_grad()
        output = model.forward(x, std_params)[:batch_size]
        losses = criterion(output, y)

        saved_var = dict()
        for tensor_name, tensor in model.named_parameters():
            saved_var[tensor_name] = torch.zeros_like(tensor)

        for j in losses:
            j.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip, norm_type=2)
            for tensor_name, tensor in model.named_parameters():
                new_grad = tensor.grad
                saved_var[tensor_name].add_(new_grad)
            optimizer.zero_grad()

        for tensor_name, tensor in model.named_parameters():
            tensor.grad = saved_var[tensor_name] / losses.shape[0]

        optimizer.step()
        scheduler_lr.step()


def evaluate_accuracy(model, data_loader, device, std_params, axi_x):
    model.eval()
    correct = 0
    total = 0.
    with torch.no_grad():
        for x, y in data_loader:
            _batch_size = x.shape[0]
            x = torch.cat([x.to(device), axi_x], dim=0)
            output = model(x, std_params)[:_batch_size]
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.to(device).view_as(pred)).sum().item()
            total += _batch_size
    accuracy = correct / total
    return accuracy

def main(noise_multiplier, clip, lr, batch_size, epochs, normalization_type, device):
    from .si_resnet import ResNet18

    epsilon = 0.05
    normalize = False
    noise_multiplier = 10.
    clip = 2.5
    lr = 0.12
    batch_size = 16
    epochs = 40

    model = ResNet18(normalize).to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_loader, test_loader = data_loaders(batch_size)
    axi_x = axi_loader(30).to(device)
    scheduler_lr = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader),
                                                       epochs=epochs)
    for epoch in range(epochs):
        print('epoch', epoch)
        train(model, criterion, optimizer, device, train_loader, clip, scheduler_lr, noise_multiplier,
              batch_size, axi_x)

        test_accuracy = evaluate_accuracy(model, test_loader, device, 1e-6, axi_x)

        print('epoch', epoch)
        print('valid accuracy', test_accuracy)
        print('---------------------')