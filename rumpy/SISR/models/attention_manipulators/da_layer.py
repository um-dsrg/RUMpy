from torch import nn
from rumpy.SISR.models.advanced import common
import torch.nn.functional as F
from colorama import init, Fore


class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, reduction=8):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(channels_in, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, channels_out * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = common.default_conv(channels_out, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        # branch 1
        kernel = self.kernel(x[1].squeeze(-1).squeeze(-1)).view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        out = self.conv(out.view(b, -1, h, w))

        # branch 2
        out = out + self.ca(x)

        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()

        if channels_in != 256:
            print('%s The DA channel attention FC layers will be initialized with a non-standard scheme.%s' % (Fore.RED, Fore.RESET))

        if channels_in <= 20:
            reduced_kernels = 32
        else:
            reduced_kernels = channels_in // reduction

        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, reduced_kernels, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(reduced_kernels, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(x[1])

        return x[0] * att
