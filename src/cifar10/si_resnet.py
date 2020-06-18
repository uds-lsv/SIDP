# https://github.com/kuangliu/pytorch-cifar

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_si_layers import SILinear, SIConv2d


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        affine = False
        track_running_stats = False
        self.conv1 = SIConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine, track_running_stats=track_running_stats)
        self.conv2 = SIConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine, track_running_stats=track_running_stats)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SIConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, affine=affine, track_running_stats=track_running_stats)
            )

    def forward(self, z):
        x, noise_std = z
        out = F.relu(self.bn1(self.conv1((x, noise_std))))
        out = self.bn2(self.conv2((out, noise_std)))
        out += self.shortcut((x, noise_std))[0]
        out = F.relu(out)
        return out, noise_std



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, normalize=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.normalize = normalize
        self.conv1 = SIConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = SILinear(512*block.expansion, num_classes, bias=True)

        if normalize:
            self.finalbn = nn.BatchNorm1d(num_classes, affine=False, track_running_stats=False)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, noise_std):
        out = F.relu(self.bn1(self.conv1((x, noise_std))))
        out = self.layer1((out, noise_std)) # out = (tensor, noise_std)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out[0], 4)
        out = out.view(out.size(0), -1)
        out = self.linear((out, noise_std))
        if self.normalize:
            out = self.finalbn(out)
        return out

def ResNet18(normalize):
    return ResNet(BasicBlock, [2,2,2,2], normalize=normalize)
