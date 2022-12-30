import os
import csv

from torch import nn
import rumpy.SISR.models.blur_kernel_blind_sr.dasr_common as common
import torch.nn.functional as F

from rumpy.regression.models.contrastive_learning.encoding_models import Encoder
from rumpy.regression.models.contrastive_learning.moco import MoCo

# DASR code taken from https://github.com/LongguangWang/DASR


class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64 * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = common.default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        # branch 1
        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        out = self.conv(out.view(b, -1, h, w))

        # branch 2
        out = out + self.ca(x)

        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(x[1][:, :, None, None])

        return x[0] * att


class DAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DAB, self).__init__()

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu =  nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv2(out) + x[0]

        return out


class DAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(DAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DAB(conv, n_feat, kernel_size, reduction) \
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        res = x[0]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1]])
        res = self.body[-1](res)
        res = res + x[0]

        return res


class DASR(nn.Module):
    def __init__(self, scale=4, n_groups=5, n_blocks=5, n_feats=64, kernel_size=3, reduction=8,
                 conv=common.default_conv, mean_sub_style=False, **kwargs):
        super(DASR, self).__init__()

        self.n_groups = n_groups
        self.mean_sub_style = mean_sub_style
        if mean_sub_style:
            # RGB mean for DIV2K
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgb_std = (1.0, 1.0, 1.0)
            self.sub_mean = common.MeanShift(255.0, rgb_mean, rgb_std)
            self.add_mean = common.MeanShift(255.0, rgb_mean, rgb_std, 1)

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # compress
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

        # body
        modules_body = [
            DAG(common.default_conv, n_feats, kernel_size, reduction, n_blocks) \
            for _ in range(n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, k_v):
        k_v = self.compress(k_v)

        if self.mean_sub_style:
            # shift data to 0-255
            x = x * 255
            # sub mean
            x = self.sub_mean(x)

        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i]([res, k_v])
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        if self.mean_sub_style:
            # add mean
            x = self.add_mean(x)
            x = x / 255

        return x


class DASRPipeline(nn.Module):
    def __init__(self, **kwargs):
        super(DASRPipeline, self).__init__()
        # Generator
        self.G = DASR(**kwargs)
        # Encoder
        self.E = MoCo(base_encoder=Encoder)

    def forward(self, x, x_key=None, **kwargs):
        if x_key is not None:
            # degradation-aware representation learning
            fea, logits, labels = self.E(x, x_key)

            # degradation-aware SR
            sr = self.G(x, fea)

            return sr, logits, labels
        else:
            # degradation-aware representation learning
            fea = self.E(x, x)

            norm_path = os.path.normpath(kwargs['model_save_dir'])
            split_path = norm_path.split(os.sep)

            # degradation-aware SR
            sr = self.G(x, fea)

            return sr
