import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single, _pair, _triple

class SILinear(Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, noise_std = 0):
        super(SILinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.noise_std = noise_std
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, noise_std = 0):
        weight_noise = noise_std * torch.randn_like(self.weight, requires_grad=False)
        if self.bias is not None:
            bias_noise = noise_std * torch.randn_like(self.bias, requires_grad=False)
            bias = self.bias + bias_noise
        else:
            bias = None
        return F.linear(input, self.weight+weight_noise, bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class SIConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SIConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def conv2d_forward(self, input, weight, noise_std):
        weight_noise = noise_std * torch.randn_like(weight, requires_grad=False)
        if self.bias is not None:
            bias_noise = noise_std * torch.randn_like(self.bias, requires_grad=False)
            bias = self.bias + bias_noise
        else:
            bias = None

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight+weight_noise, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight+weight_noise, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input, noise_std):
        return self.conv2d_forward(input, self.weight, noise_std)


class SILSTMCell(nn.LSTMCell):

    def __init__(self, input_size, hidden_size, noise_std, bias=True):
        super().__init__(input_size, hidden_size, bias)

        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)
        self.noise_std = noise_std

    def forward(self, input, hidden=None, noise_std=0.0):
        if noise_std > 0.0:
            self.noise_std = noise_std
        weight_noise_ih = self.noise_std * torch.randn_like(self.weight_ih, requires_grad=False)
        weight_noise_hh = self.noise_std * torch.randn_like(self.weight_hh, requires_grad=False)
        bias_noise_ih = self.noise_std * torch.randn_like(self.bias_ih, requires_grad=False)
        bias_noise_hh = self.noise_std * torch.randn_like(self.bias_hh, requires_grad=False)

        self.check_forward_input(input)
        if hidden is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden
        self.check_forward_hidden(input, hx, '[0]')
        self.check_forward_hidden(input, cx, '[1]')

        gates = self.ln_ih(F.linear(input, self.weight_ih + weight_noise_ih, self.bias_ih + bias_noise_ih)) \
                + self.ln_hh(F.linear(hx, self.weight_hh + weight_noise_hh, self.bias_hh + bias_noise_hh))
        i, f, o = gates[:, :(3 * self.hidden_size)].sigmoid().chunk(3, 1)
        g = gates[:, (3 * self.hidden_size):].tanh()

        cy = (f * cx) + (i * g)
        hy = o * self.ln_ho(cy).tanh()
        return hy, cy


class SILSTM(nn.Module):

    def __init__(self, input_size, hidden_size, noise_std, num_layers=1, bias=True, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.noise_std = noise_std

        # print("LayerNormLSTMNoisy", self.noise_std)
        num_directions = 2 if bidirectional else 1
        self.hidden0 = nn.ModuleList([
            SILSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                                   hidden_size=hidden_size, noise_std=self.noise_std, bias=bias)
            for layer in range(num_layers)
        ])

        if self.bidirectional:
            self.hidden1 = nn.ModuleList([
                SILSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                                       hidden_size=hidden_size, noise_std=self.noise_std, bias=bias)
                for layer in range(num_layers)
            ])

    def forward(self, input, noise_std=0.0, hidden=None):
        seq_len, batch_size, hidden_size = input.size()  # supports TxNxH only
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            hx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
            cx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        ht = [[None, ] * (self.num_layers * num_directions)] * seq_len
        ct = [[None, ] * (self.num_layers * num_directions)] * seq_len

        if self.bidirectional:
            xs = input
            for l, (layer0, layer1) in enumerate(zip(self.hidden0, self.hidden1)):
                l0, l1 = 2 * l, 2 * l + 1
                h0, c0, h1, c1 = hx[l0], cx[l0], hx[l1], cx[l1]
                for t, (x0, x1) in enumerate(zip(xs, reversed(xs))):
                    ht[t][l0], ct[t][l0] = layer0(x0, (h0, c0), noise_std)
                    h0, c0 = ht[t][l0], ct[t][l0]
                    t = seq_len - 1 - t
                    ht[t][l1], ct[t][l1] = layer1(x1, (h1, c1), noise_std)
                    h1, c1 = ht[t][l1], ct[t][l1]
                xs = [torch.cat((h[l0], h[l1]), dim=1) for h in ht]
            y = torch.stack(xs)
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])
        else:
            h, c = hx, cx
            for t, x in enumerate(input):
                for l, layer in enumerate(self.hidden0):
                    ht[t][l], ct[t][l] = layer(x, (h[l], c[l]), noise_std)
                    x = ht[t][l]
                h, c = ht[t], ct[t]
            y = torch.stack([h[-1] for h in ht])
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])


        return y, (hy, cy)

