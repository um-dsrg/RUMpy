# All code here modified from https://github.com/JingyunLiang/MANet
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from collections import OrderedDict


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# --------------------------------------------
# MAConv and MABlock for MANet
# --------------------------------------------
class MAConv(nn.Module):
    ''' Mutual Affine Convolution (MAConv) layer '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, split=2, reduction=2):
        super(MAConv, self).__init__()
        assert split >= 2, 'Num of splits should be larger than one'

        self.num_split = split
        splits = [1 / split] * split
        self.in_split, self.in_split_rest, self.out_split = [], [], []

        for i in range(self.num_split):
            in_split = round(in_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.in_split)
            in_split_rest = in_channels - in_split
            out_split = round(out_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.out_split)

            self.in_split.append(in_split)
            self.in_split_rest.append(in_split_rest)
            self.out_split.append(out_split)

            setattr(self, 'fc{}'.format(i), nn.Sequential(*[
                nn.Conv2d(in_channels=in_split_rest, out_channels=int(in_split_rest // reduction),
                          kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(in_split_rest // reduction), out_channels=in_split * 2,
                          kernel_size=1, stride=1, padding=0, bias=True),
            ]))
            setattr(self, 'conv{}'.format(i), nn.Conv2d(in_channels=in_split, out_channels=out_split,
                                                        kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, input):
        input = torch.split(input, self.in_split, dim=1)
        output = []

        for i in range(self.num_split):
            scale, translation = torch.split(getattr(self, 'fc{}'.format(i))(torch.cat(input[:i] + input[i + 1:], 1)),
                                             (self.in_split[i], self.in_split[i]), dim=1)
            output.append(getattr(self, 'conv{}'.format(i))(input[i] * torch.sigmoid(scale) + translation))

        return torch.cat(output, 1)


class MABlock(nn.Module):
    ''' Residual block based on MAConv '''
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True,
                 split=2, reduction=2):
        super(MABlock, self).__init__()

        self.res = nn.Sequential(*[
            MAConv(in_channels, in_channels, kernel_size, stride, padding, bias, split, reduction),
            nn.ReLU(inplace=True),
            MAConv(in_channels, out_channels, kernel_size, stride, padding, bias, split, reduction),
        ])

    def forward(self, x):
        return x + self.res(x)


# ------------------------------------------------------
# MANet and its combinations with non-blind SR
# ------------------------------------------------------

class MANet(nn.Module):
    ''' Network of MANet'''
    def __init__(self, in_nc=3, kernel_size=21, nc=[128, 256], nb=1, split=2, scale=4):
        super(MANet, self).__init__()
        self.kernel_size = kernel_size

        self.m_head = nn.Conv2d(in_channels=in_nc, out_channels=nc[0], kernel_size=3, padding=1, bias=True)
        self.m_down1 = sequential(*[MABlock(nc[0], nc[0], bias=True, split=split) for _ in range(nb)],
                                  nn.Conv2d(in_channels=nc[0], out_channels=nc[1], kernel_size=2, stride=2, padding=0,
                                            bias=True))

        self.m_body = sequential(*[MABlock(nc[1], nc[1], bias=True, split=split) for _ in range(nb)])

        self.m_up1 = sequential(nn.ConvTranspose2d(in_channels=nc[1], out_channels=nc[0],
                                                   kernel_size=2, stride=2, padding=0, bias=True),
                                *[MABlock(nc[0], nc[0], bias=True, split=split) for _ in range(nb)])
        self.m_tail = nn.Conv2d(in_channels=nc[0], out_channels=kernel_size ** 2, kernel_size=3, padding=1, bias=True)

        self.scale = scale
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x = self.m_body(x2)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        x = x[..., :h, :w]

        x = self.softmax(x)

        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')  # final interpolation step to extend kernels to same size as HR image

        return x

# TODO: continue figuring out how MaNet works, and what to do to fix it up...
