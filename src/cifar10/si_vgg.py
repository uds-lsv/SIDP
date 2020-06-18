# https://github.com/kuangliu/pytorch-cifar

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg_si_layers import SILinear, SIConv2d, SIBatchNorm2d, SIReLU, SIMaxPool2d, SIAvgPool2d


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, normalize=None):
        super(VGG, self).__init__()
        self.normalize = normalize
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = SILinear(512, 10, bias=True)
        if normalize:
            self.finalbn = nn.BatchNorm1d(10, affine=False, track_running_stats=False)

    def forward(self, x, noise_std):
        out = self.features((x, noise_std))[0]
        out = out.view(out.size(0), -1)
        out = self.classifier((out, noise_std))
        if self.normalize:
            out = self.finalbn(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [SIMaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [SIConv2d(in_channels, x, kernel_size=3, padding=1),
                           SIBatchNorm2d(x, affine=False, track_running_stats=False),
                           SIReLU(inplace=True)]
                in_channels = x
        layers += [SIAvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


# net = VGG('VGG16', noise_std=.01, normalize=False)
# print(net)
