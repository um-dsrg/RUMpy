import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, dropdown_q=None):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )
        if dropdown_q is not None:
            self.drop_mlp = nn.Sequential(
                nn.Linear(256, 64),
                nn.LeakyReLU(0.1, True),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.1, True),
                nn.Linear(32, dropdown_q),
            )
            self.dropdown = True
        else:
            self.dropdown = False

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        out_dict = {'q': out}
        if self.dropdown:
            out_dict['dropdown_q'] = self.drop_mlp(out)

        return fea, out_dict


class ChannelAttention(nn.Module):
    """
    Taken from GitHub: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Taken from GitHub: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(in_planes=in_planes, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x_ca = x * self.ca(x)
        x_ca_sa = x_ca * self.sa(x_ca)

        return x_ca_sa


class IDMN(nn.Module):
    """
    Implicit Degradation Modeling Network

    This module was defined in https://www.sciencedirect.com/science/article/pii/S0950705122004774 and uses
    a similar encoder to DASR but introduces a CBAM module and removes BatchNorm and LReLU.
    """

    def __init__(self):
        super(IDMN, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            CBAM(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )

        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out

