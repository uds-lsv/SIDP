import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dp_layers.dp_layers import SILinear, SIConv2d


class LeNet5(nn.Module):
    def __init__(self, normalization_type):
        super(LeNet5, self).__init__()
        self.normalization_type = normalization_type
        self.conv1 =SIConv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = SIConv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = SILinear(16*5*5, 120)
        self.fc2 = SILinear(120, 84)
        self.fc3 = SILinear(84, 10)

        if normalization_type == 'Layer':
            self.n1 = nn.LayerNorm([6, 28, 28], elementwise_affine=False)
            self.n2 = nn.LayerNorm([16, 10, 10], elementwise_affine=False)
            self.n3 = nn.LayerNorm(120, elementwise_affine=False)
            self.n4 = nn.LayerNorm(84, elementwise_affine=False)
        elif normalization_type == 'Batch':
            self.n1 = nn.BatchNorm2d(6, affine=False, track_running_stats=False)
            self.n2 = nn.BatchNorm2d(16, affine=False, track_running_stats=False)
            self.n3 = nn.BatchNorm1d(120, affine=False, track_running_stats=False)
            self.n4 = nn.BatchNorm1d(84, affine=False, track_running_stats=False)


    def forward(self, x, noise_std):
        x = F.relu(self.conv1(x, noise_std))

        if self.normalization_type is not None:
            x = self.n1(x)

        x = self.max_pool_1(x)
        x = F.relu(self.conv2(x, noise_std))

        if self.normalization_type is not None:
            x = self.n2(x)

        x = self.max_pool_2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x, noise_std))

        if self.normalization_type is not None:
            x = self.n3(x)

        x = F.relu(self.fc2(x, noise_std))
        if self.normalization_type is not None:
            x = self.n4(x)

        x = self.fc3(x, noise_std)

        return x

