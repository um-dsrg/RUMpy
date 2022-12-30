import math
import numbers
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import rumpy.SISR.models.advanced.SRMD_blocks as B
from rumpy.SISR.models.advanced import common
from rumpy.SISR.models.advanced.HAN_blocks import CSAM_Module, LAM_Module
from rumpy.SISR.models.advanced.SAN_blocks import LSRAG, Nonlocal_CA
from rumpy.SISR.models.advanced.ELAN_blocks import ELAB, MeanShift

# Channel Attention (CA) Layer


class CALayer(nn.Module):
    """
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """

    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

    def forensic(self, x):

        inner_forensic_data = {}

        y = self.avg_pool(x)
        inner_forensic_data['inner_vector'] = self.conv_du[1](self.conv_du[0](y)).cpu().data.numpy().squeeze()
        y = self.conv_du(y)

        inner_forensic_data['mask_multiplier'] = y.cpu().data.numpy().squeeze()

        return x * y, inner_forensic_data


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    """
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """

    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

    def forensic(self, x):
        res = x
        conv_data = []
        for module in self.body:
            if isinstance(module, CALayer):
                res, forensic_data = module.forensic(res)
            else:
                res = module.forward(res)
                if isinstance(module, nn.Conv2d):
                    conv_data.append(module.weight.detach().cpu().numpy().flatten())

        forensic_data['conv_flat'] = np.hstack(np.array(conv_data))
        forensic_data['pre-residual'] = res
        forensic_data['pre-residual-flat'] = res.cpu().numpy().flatten()
        res += x
        forensic_data['post-residual'] = res
        forensic_data['post-residual-flat'] = res.cpu().numpy().flatten()
        return res, forensic_data


# Residual Group (RG)
class ResidualGroup(nn.Module):
    """
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """

    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

    def forensic(self, x):
        res = x
        forensic_data = []
        for module in self.body:
            if isinstance(module, RCAB):
                res, RCAB_data = module.forensic(res)
                forensic_data.append(RCAB_data)
            else:
                res = module.forward(res)
        res += x
        return res, forensic_data


# Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    """
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """

    def __init__(self, n_resblocks=20, n_resgroups=10, n_feats=64, in_feats=3, out_feats=3, scale=4, reduction=16,
                 res_scale=1.0, **kwargs):
        super(RCAN, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        # define head module
        modules_head = [common.default_conv(in_feats, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(common.default_conv, n_feats, kernel_size, reduction,
                          act=act, res_scale=res_scale, n_resblocks=n_resblocks) for _ in range(n_resgroups)]

        modules_body.append(common.default_conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(common.default_conv, scale, n_feats, act=False),
            common.default_conv(n_feats, out_feats, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

    def forensic(self, x, *args, **kwargs):
        x = self.head(x)
        data = OrderedDict()
        res = x
        for index, module in enumerate(self.body):
            if isinstance(module, ResidualGroup):
                res, res_forensic_data = module.forensic(res)
                for rcab_index, rcab_forensic_data in enumerate(res_forensic_data):
                    data['R%d.C%d' % (index, rcab_index)] = rcab_forensic_data
            else:
                res = module.forward(res)
        res += x
        x = self.tail(res)
        return x, data

    def reset_parameters(self):
        # TODO: Find out how to do this!
        pass


class EDSR(nn.Module):
    """
    EDSR - conv net using multiple residual connections, and taking a full-RGB input image.
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    Standard EDSR has the following parameters:
    net_features = 256, num_blocks = 32, res_scale = 0.1.  For scale factors above 2, the original paper
    first pre-trained an X2 model, before re-training for X4 performance.
    """

    def __init__(self, in_features=3, out_features=3, net_features=64, num_blocks=16, scale=4, res_scale=0.1):
        super(EDSR, self).__init__()

        n_resblocks = num_blocks
        n_feats = net_features
        kernel_size = 3
        act = nn.ReLU(True)

        # define head module
        m_head = [common.default_conv(in_features, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                common.default_conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(common.default_conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(common.default_conv, scale, n_feats),
            common.default_conv(n_feats, out_features, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

    # @staticmethod
    # def weight_reset(m):
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         m.reset_parameters()
    #     else:
    #         print()
    #     if isinstance(m, nn.Sequential):
    #         m.apply(EDSR.weight_reset)

    def reset_parameters(self):
        # TODO: Find out how to do this!
        pass
        # self.head.apply(self.weight_reset)
        # self.body.apply(self.weight_reset)
        # self.tail.apply(self.weight_reset)


class SAN(nn.Module):
    """
    Based on implementation in https://github.com/daitao/SAN
    """

    def __init__(self, n_resgroups=20, n_resblocks=10, n_feats=64, reduction=16, scale=4, rgb_range=255, n_colors=3,
                 res_scale=1, conv=common.default_conv):
        super(SAN, self).__init__()
        n_resgroups = n_resgroups
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        reduction = reduction
        scale = scale
        act = nn.ReLU(inplace=True)

        # self.soca= SOCA(n_feats, reduction=reduction)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        ##
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = 0.2
        self.n_resgroups = n_resgroups
        self.RG = nn.ModuleList([LSRAG(conv, n_feats, kernel_size, reduction,
                                       act=act, res_scale=res_scale, n_resblocks=n_resblocks) for _ in range(n_resgroups)])
        self.conv_last = conv(n_feats, n_feats, kernel_size)

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        self.non_local = Nonlocal_CA(in_feat=n_feats, inter_feat=n_feats // 8,
                                     reduction=8, sub_sample=False, bn_layer=False)

        self.head = nn.Sequential(*modules_head)
        # self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)

        return nn.ModuleList(layers)

    def forward(self, x):

        x = self.head(x)

        # add nonlocal
        xx = self.non_local(x)

        # share-source skip connection
        residual = xx

        # share-source residual gruop
        for i, l in enumerate(self.RG):
            xx = l(xx) + self.gamma*residual

        # add nonlocal
        res = self.non_local(xx)
        res = res + x

        x = self.tail(res)

        return x


class HAN(nn.Module):
    # Based on implementation in https://github.com/wwlCape/HAN
    def __init__(self, n_resgroups=10, n_resblocks=20, n_feats=64, reduction=16,
                 scale=4, n_colors=3, res_scale=1.0, conv=common.default_conv):
        super(HAN, self).__init__()

        n_resgroups = n_resgroups
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        reduction = reduction
        scale = scale
        act = nn.ReLU(True)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks)
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.csa = CSAM_Module(n_feats)
        self.la = LAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats*11, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats*2, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        x = self.head(x)
        res = x

        for name, midlayer in self.body._modules.items():
            res = midlayer(res)
            if name == '0':
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1), res1], 1)

        out1 = res

        res = self.la(res1)
        out2 = self.last_conv(res)

        out1 = self.csa(out1)
        out = torch.cat([out1, out2], 1)
        res = self.last(out)

        res += x

        x = self.tail(res)

        return x


class SRMD(nn.Module):
    """
    Based on implementation in https://github.com/cszn/KAIR/
    Defaults to noise-free model here.
    """

    def __init__(self, in_nc=18, out_nc=3, nc=128, nb=12, scale=4, act_mode='R', upsample_mode='pixelshuffle', **kwargs):
        """
        # ------------------------------------
        in_nc: channel number of input, default: 3+15
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        upscale: scale factor
        act_mode: batch norm + activation function; 'BR' means BN+ReLU
        upsample_mode: default 'pixelshuffle' = conv + pixelshuffle
        # ------------------------------------
        """
        super(SRMD, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = upsample_block(nc, out_nc, mode=str(scale), bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    #    def forward(self, x, k_pca):
    #        m = k_pca.repeat(1, 1, x.size()[-2], x.size()[-1])
    #        x = torch.cat((x, m), 1)
    #        x = self.body(x)

    def forward(self, x):

        x = self.model(x)

        return x


class ELAN(nn.Module):
    """
    Based on the implementation from: https://github.com/xindongzhang/ELAN/blob/main/models/elan_network.py
    """
    def __init__(self, scale=4,
                 colors=3, window_sizes=[4, 8, 16],
                 m_elan=36, c_elan=180,
                 n_share=0, r_expand=2,
                 apply_mean_shift=True,
                 rgb_range=1.0, **kwargs):
        super(ELAN, self).__init__()

        self.scale = scale
        self.colors = colors
        self.window_sizes = window_sizes
        self.m_elan = m_elan
        self.c_elan = c_elan
        self.n_share = n_share
        self.r_expand = r_expand
        self.apply_mean_shift = apply_mean_shift

        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [nn.Conv2d(self.colors, self.c_elan, kernel_size=3, stride=1, padding=1)]

        # define body module
        m_body = []
        for i in range(self.m_elan // (1+self.n_share)):
            if (i+1) % 2 == 1:
                m_body.append(
                    ELAB(
                        self.c_elan, self.c_elan, self.r_expand, 0,
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:
                m_body.append(
                    ELAB(
                        self.c_elan, self.c_elan, self.r_expand, 1,
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
        # define tail module
        m_tail = [
            nn.Conv2d(self.c_elan, self.colors*self.scale*self.scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        if self.apply_mean_shift:
            x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.tail(res)

        if self.apply_mean_shift:
            x = self.add_mean(x)

        return x[:, :, 0:H*self.scale, 0:W*self.scale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
