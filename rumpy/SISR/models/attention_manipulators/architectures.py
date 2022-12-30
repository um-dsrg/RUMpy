import math
from collections import OrderedDict

import numpy as np
import torch
from rumpy.SISR.models.advanced.architectures import common
from rumpy.SISR.models.advanced.ELAN_blocks import GMSA, LFE, MeanShift
from rumpy.SISR.models.advanced.HAN_blocks import CSAM_Module, LAM_Module
from rumpy.SISR.models.advanced.SAN_blocks import Nonlocal_CA, LSRAG
from rumpy.SISR.models.attention_manipulators.da_layer import DA_conv
from rumpy.SISR.models.attention_manipulators.dgfmb_layer import DGFMBLayer
from rumpy.SISR.models.attention_manipulators.q_layer import ParaCALayer
from rumpy.SISR.models.attention_manipulators.qsan_blocks import QLSRAG
from rumpy.SISR.models.SFTMD_variants.architectures import StandardSft
from rumpy.SISR.models.non_blind_gan_models.generators import (ResidualDenseBlock, pixel_unshuffle)
from torch import nn
from torch.nn import functional as F


class PALayer(nn.Module):
    # adapted from https://github.com/zhilin007/FFA-Net/blob/master/net/models/FFA.py
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

    def forensic(self, x):
        y = self.pa(x)
        return x * y, y.cpu().data.numpy().squeeze()


# Channel Attention (CA) Layer
class QCALayer(nn.Module):
    """
    Combined channel-attention and meta-attention layer.  Diverse style choices available.
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """

    def __init__(self, channel, style, reduction=16, num_metadata=1):
        """
        :param channel:  Network feature map channel count.
        :param style: Type of attention to use.  Options are:
        modulate:  Normal channel attention occurs, but meta-vector is multiplied with the final attention
        vector prior to network modulation.
        mini_concat:  Concatenate meta-vector with inner channel attention vector.
        max_concat:  Concatenate meta-vector with feature map aggregate, straight after average pooling.
        softmax:  Implements max_concat, but also applies softmax after the final FC layer.
        extended_attention: Splits attention into four layers, and adds metadata vector in second layer.
        standard:  Do not introduce any metadata.
        :param reduction: Level of downscaling to use for inner channel attention vector.
        :param num_metadata: Expected metadata input size.
        """
        super(QCALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight

        if reduction < 16:
            raise RuntimeError('Using an extreme channel attention reduction value')

        if style == 'modulate' or style == 'mini_concat' or style == 'standard':
            channel_in = channel
        else:
            channel_in = channel + num_metadata

        channel_reduction = channel // reduction

        if style == 'modulate' or style == 'max_concat' or style == 'softmax' or style == 'standard':
            self.conv_du = nn.Sequential(
                nn.Conv2d(channel_in, channel_reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel_reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )
        elif style == 'mini_concat':
            self.pre_concat = nn.Conv2d(channel_in, channel_reduction, 1, padding=0, bias=True)
            self.conv_du = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(channel_reduction + num_metadata, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )
        elif style == 'extended_attention':

            channel_fractions = [(channel_in, channel // 2),
                                 (channel // 2 + num_metadata, channel // 4),
                                 (channel // 4 + num_metadata, channel_reduction)]
            self.feature_convs = nn.ModuleList()
            for (inp, outp) in channel_fractions:
                self.feature_convs.append(
                    nn.Sequential(
                        nn.Conv2d(inp, outp, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True)
                    )
                )
            self.final_conv = nn.Sequential(
                nn.Conv2d(channel_reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )

        if style == 'softmax':
            self.softmax = nn.Softmax(dim=1)

        self.style = style

    def forward(self, x, attributes):

        y = self.avg_pool(x)
        if self.style == 'modulate':
            y = self.conv_du(y) * attributes
        elif self.style == 'max_concat':
            y = self.conv_du(torch.cat((y, attributes), dim=1))
        elif self.style == 'mini_concat':
            y = self.pre_concat(y)
            y = self.conv_du(torch.cat((y, attributes), dim=1))
        elif self.style == 'extended_attention':
            for conv_section in self.feature_convs:
                y = conv_section(torch.cat((y, attributes), dim=1))
            y = self.final_conv(y)
        elif self.style == 'softmax':
            y = self.conv_du(torch.cat((y, attributes), dim=1))
            y = self.softmax(y)
        elif self.style == 'standard':
            y = self.conv_du(y)
        else:
            raise NotImplementedError

        return x * y

    def forensic(self, x, attributes):
        inner_forensic_data = {}
        y = self.avg_pool(x)
        if self.style == 'standard':
            inner_forensic_data['inner_vector'] = self.conv_du[1](self.conv_du[0](y)).cpu().data.numpy().squeeze()
            y = self.conv_du(y)
        else:
            inner_forensic_data['inner_vector'] = self.conv_du[1](
                self.conv_du[0](torch.cat((y, attributes), dim=1))).cpu().data.numpy().squeeze()
            y = self.conv_du(torch.cat((y, attributes), dim=1))

        inner_forensic_data['mask_multiplier'] = y.cpu().data.numpy().squeeze()

        return x * y, inner_forensic_data


# Residual Channel Attention Block (RCAB)
class QRCAB(nn.Module):
    """
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """

    def __init__(
            self, conv, n_feat, kernel_size, reduction, style='modulate', pa=False, q_layer=False,
            dgfmb_layer=False, sft_layer=False, da_conv_layer=False, bias=True, bn=False, act=nn.ReLU(True),
            res_scale=1, num_metadata=1, num_layers_in_q_layer=2, num_layers_in_dgfmb_layer=2,
            use_dgfmb_reduction=True):

        super(QRCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        self.final_body = QCALayer(channel=n_feat, reduction=reduction, style=style, num_metadata=num_metadata)
        self.pa = pa
        self.q_layer = q_layer
        self.dgfmb_layer = dgfmb_layer
        self.da_conv_layer = da_conv_layer
        self.sft_layer = sft_layer
        if pa:
            self.pa_node = PALayer(channel=n_feat)
        if q_layer:
            self.q_node = ParaCALayer(network_channels=n_feat, num_metadata=num_metadata, nonlinearity=True,
                                      num_layers=num_layers_in_q_layer)
        if dgfmb_layer:
            self.dgfmb_node = DGFMBLayer(num_channels=n_feat,
                                         degradation_full_dim=num_metadata,
                                         num_layers=num_layers_in_dgfmb_layer,
                                         use_linear=False,
                                         use_reduction=use_dgfmb_reduction)  # Should we use FC or the 'Conv2d trick'?
        if da_conv_layer:
            self.da_node = DA_conv(channels_in=num_metadata, channels_out=n_feat)
        if sft_layer:
            self.sft_node = StandardSft(nf=n_feat, para=num_metadata)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x[0])
        res = self.final_body(res, x[1])
        if self.pa:
            res = self.pa_node(res)
        if self.q_layer:
            if isinstance(x[1], list):
                for vec_index, (vector, vector_used) in enumerate(x[1]):
                    if not vector_used:
                        x[1][vec_index][1] = True
                        final_vector = vector
                        break
            else:
                final_vector = x[1]
            res = self.q_node(res, final_vector)
        if self.dgfmb_layer:
            res = self.dgfmb_node(res, x[1])
        if self.da_conv_layer:
            res = self.da_node((res, x[1]))
        if self.sft_layer:
            res = self.sft_node(res, x[1])

        res += x[0]
        return res, x[1]

    def forensic(self, x, qpi):

        res = self.body(x)
        conv_data = []
        for module in self.body:
            if isinstance(module, nn.Conv2d):
                conv_data.append(module.weight.detach().cpu().numpy().flatten())

        res, forensic_data = self.final_body.forensic(res, qpi)
        if self.pa:
            res, forensic_pa = self.pa_node.forensic(res)
            forensic_data['pixel_attention_map'] = forensic_pa
        if self.q_layer:
            res, _, _, meta_attention = self.q_node.forensic(res, qpi)
            forensic_data['meta_attention_map'] = meta_attention

        forensic_data['pre-residual'] = res
        forensic_data['pre-residual-flat'] = res.cpu().numpy().flatten()
        res += x
        forensic_data['post-residual'] = res
        forensic_data['post-residual-flat'] = res.cpu().numpy().flatten()
        forensic_data['conv_flat'] = np.hstack(np.array(conv_data))
        return res, forensic_data


# Residual Group (RG)
class QResidualGroup(nn.Module):
    """
    Based on implementation in https://github.com/thstkdgus35/EDSR-PyTorch
    """

    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, style, num_metadata,
                 pa, q_layer, dgfmb_layer, da_conv_layer,
                 num_q_layers, num_layers_in_q_layer,
                 sft_layer, num_sft_layers,
                 num_dgfmb_layers, num_layers_in_dgfmb_layer, use_dgfmb_reduction,
                 num_da_conv_layers):
        super(QResidualGroup, self).__init__()
        modules_body = []

        for index in range(n_resblocks):
            if num_q_layers is None or index < num_q_layers:
                q_in = q_layer
            else:
                q_in = False

            if num_dgfmb_layers is None or index < num_dgfmb_layers:
                dgfmb_in = dgfmb_layer
            else:
                dgfmb_in = False

            if num_da_conv_layers is None or index < num_da_conv_layers:
                da_conv_in = da_conv_layer
            else:
                da_conv_in = False

            if num_sft_layers is None or index < num_sft_layers:
                sft_in = sft_layer
            else:
                sft_in = False

            modules_body.append(QRCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False,
                                      act=act, res_scale=res_scale, style=style,
                                      pa=pa, q_layer=q_in, dgfmb_layer=dgfmb_in, da_conv_layer=da_conv_in,
                                      num_metadata=num_metadata,
                                      num_layers_in_q_layer=num_layers_in_q_layer,
                                      sft_layer=sft_in,
                                      num_layers_in_dgfmb_layer=num_layers_in_dgfmb_layer,
                                      use_dgfmb_reduction=use_dgfmb_reduction))

        self.final_body = conv(n_feat, n_feat, kernel_size)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res, _ = self.body(x)
        res = self.final_body(res)
        res += x[0]
        return res, x[1]

    def forensic(self, x, qpi):
        res = x
        forensic_data = []
        for module in self.body:
            res, RCAB_data = module.forensic(res, qpi)
            forensic_data.append(RCAB_data)
        res = self.final_body(res)
        res += x
        return res, forensic_data


class QRCAN(nn.Module):
    def __init__(self, n_resblocks=20, n_resgroups=10, n_feats=64, in_feats=3, out_feats=3, scale=4, reduction=16,
                 res_scale=1.0, style='modulate', num_metadata=1, include_pixel_attention=False,
                 selective_meta_blocks=None,
                 include_q_layer=False, num_q_layers_inner_residual=None, num_layers_in_q_layer=2,
                 include_sft_layer=False, num_sft_layers_inner_residual=None,
                 include_dgfmb_layer=False, num_dgfmb_layers_inner_residual=None, num_layers_in_dgfmb_layer=2,
                 use_dgfmb_reduction=True, use_dgfmb_outer_reduction=False,
                 include_da_conv_layer=False, num_da_conv_layers_inner_residual=None, staggered_encoding=False,
                 **kwargs):
        """
        Main QRCAN architecture - make sure to check handler docstring for more info on this model.
        :param n_resblocks: Number of residual blocks (within each residual group).
        :param n_resgroups: Number of residual groups (following original paper nomenclature).
        :param n_feats: Number of channels within network conv layers.
        :param in_feats: Input features (if RGB, leave at 3 channels).
        :param out_feats: Output features (if RGB, leave at 3 channels).
        :param scale: SR scale.
        :param reduction: Magnitude of reduction for channel attention (e.g. for a reduction of 16 and a n_feats size of 64,
         internal channel vector will have a size of 4.
        :param res_scale: Scale factor to apply to each residual group.  By default not used (set to 1).
        :param style: Channel attention style, if combining channel attention with meta-attention.
        :param num_metadata: If using meta-attention, indicate the expected metadata vector size here.
        :param include_pixel_attention: Set to true to include pixel attention after each residual block.
        :param selective_meta_blocks:  *Check QRCAN handler for more info*
        :param include_q_layer: *Check QRCAN handler for more info*
        :param num_q_layers_inner_residual: *Check QRCAN handler for more info*
        :param num_layers_in_q_layer: Number of layers in each q-layer block.
        :param include_dgfmb_layer: *Same as for q layer but with dgfmb*
        :param num_dgfmb_layers_inner_residual: *Same as for q layer but with dgfmb*
        :param num_layers_in_dgfmb_layer: Number of layers in each dgfmb-layer block.
        :param use_dgfmb_reduction: Set to true to use the reduction layer in the dgfmb-layer block.
        :param include_sft_layer: *Same as for q layer but with SFT*
        :param num_sft_layers_inner_residual: *Same as for q layer but with SFT*
        :param include_da_conv_layer: *Same as for q layer but with da conv layer*
        :param num_da_conv_layers_inner_residual: *Same as for q layer but with da conv*
        """
        super(QRCAN, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        if style != 'standard' and staggered_encoding:
            raise RuntimeError('QRCAN must be set to standard for staggered encoding to work.')
        self.style = style
        self.staggered_encoding = staggered_encoding

        self.metadata_reduction = nn.Sequential(nn.Identity())

        if use_dgfmb_reduction == True and use_dgfmb_outer_reduction == True:
            raise RuntimeError(
                'Cannot have both reduction modes, choose between use_dgfmb_reduction (per-block reduction) or use_dgfmb_outer_reduction (single outer reduction).')

        if use_dgfmb_outer_reduction:
            degradation_full_dim = num_metadata
            degradation_reduced_dim = 64

            # NEW METADATA SIZE WITH OUTER REDUCER
            num_metadata = degradation_reduced_dim

            self.metadata_reduction = nn.Sequential(
                nn.Conv2d(degradation_full_dim, degradation_reduced_dim, 1, padding=0, bias=True)
            )

        # define head module
        modules_head = [common.default_conv(in_feats, n_feats, kernel_size)]

        # define body module
        if selective_meta_blocks is None:
            modules_body = [
                QResidualGroup(common.default_conv, n_feats, kernel_size, reduction, style=style,
                               num_metadata=num_metadata, pa=include_pixel_attention,
                               q_layer=include_q_layer, dgfmb_layer=include_dgfmb_layer,
                               sft_layer=include_sft_layer,
                               da_conv_layer=include_da_conv_layer,
                               act=act, res_scale=res_scale, n_resblocks=n_resblocks,
                               num_q_layers=num_q_layers_inner_residual, num_layers_in_q_layer=num_layers_in_q_layer,
                               num_dgfmb_layers=num_dgfmb_layers_inner_residual,
                               num_sft_layers=num_sft_layers_inner_residual,
                               num_layers_in_dgfmb_layer=num_layers_in_dgfmb_layer,
                               use_dgfmb_reduction=use_dgfmb_reduction,
                               num_da_conv_layers=num_da_conv_layers_inner_residual) for _ in range(n_resgroups)]
        else:
            modules_body = []

            for index in range(n_resgroups):
                if selective_meta_blocks[index]:
                    include_q = include_q_layer
                    include_dgfmb = include_dgfmb_layer
                    include_da_conv = include_da_conv_layer
                    include_sft = include_sft_layer
                else:
                    include_q = False
                    include_dgfmb = False
                    include_da_conv = False
                    include_sft = False

                modules_body.append(
                    QResidualGroup(common.default_conv, n_feats, kernel_size, reduction, style=style,
                                   num_metadata=num_metadata, pa=include_pixel_attention,
                                   q_layer=include_q, dgfmb_layer=include_dgfmb, da_conv_layer=include_da_conv,
                                   sft_layer=include_sft,
                                   act=act, res_scale=res_scale, n_resblocks=n_resblocks,
                                   num_q_layers=num_q_layers_inner_residual,
                                   num_layers_in_q_layer=num_layers_in_q_layer,
                                   num_dgfmb_layers=num_dgfmb_layers_inner_residual,
                                   num_layers_in_dgfmb_layer=num_layers_in_dgfmb_layer,
                                   num_sft_layers=num_sft_layers_inner_residual,
                                   use_dgfmb_reduction=use_dgfmb_reduction,
                                   num_da_conv_layers=num_da_conv_layers_inner_residual))

        self.final_body = common.default_conv(n_feats, n_feats, kernel_size)

        # define tail module
        modules_tail = [
            common.Upsampler(common.default_conv, scale, n_feats, act=False),
            common.default_conv(n_feats, out_feats, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, metadata):

        if self.staggered_encoding:
            metadata = [[vector, False] for vector in metadata]

        metadata = self.metadata_reduction(metadata)

        x = self.head(x)
        res, *_ = self.body((x, metadata))
        res = self.final_body(res)
        res += x
        x = self.tail(res)

        return x

    def forensic(self, x, qpi, *args, **kwargs):
        x = self.head(x)
        data = OrderedDict()
        res = x
        for index, module in enumerate(self.body):
            res, res_forensic_data = module.forensic(res, qpi)
            for rcab_index, rcab_forensic_data in enumerate(res_forensic_data):
                data['R%d.C%d' % (index, rcab_index)] = rcab_forensic_data
        res = self.final_body(res)
        res += x
        x = self.tail(res)
        return x, data


class ParamResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, n_params, kernel_size, act=nn.ReLU(True),
            bias=True, res_scale=1.0, q_layer_nonlinearity=False, add_q_layer=None, num_layers=2):

        super(ParamResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)

        self.add_q_layer = add_q_layer

        if self.add_q_layer:
            self.attention_layer = ParaCALayer(n_feats, n_params, nonlinearity=q_layer_nonlinearity,
                                               num_layers=num_layers)

        self.res_scale = res_scale

    def forward(self, x):

        params = x[1]
        res = self.body(x[0])
        res = res.mul(self.res_scale)
        if self.add_q_layer:
            res = self.attention_layer(res, x[1])
        res += x[0]

        return res, params


class QEDSR(nn.Module):
    """
    modified EDSR to allow insertion of meta-attention.  Refer to original EDSR for info on function inputs.
    """

    def __init__(self,
                 in_features=3, out_features=3, num_features=64, input_para=1,
                 num_blocks=16, scale=4, res_scale=0.1, q_layer_nonlinearity=False,
                 selective_meta_blocks=None, num_layers=2, **kwargs):
        super(QEDSR, self).__init__()

        n_resblocks = num_blocks
        n_feats = num_features
        kernel_size = 3

        if selective_meta_blocks == 'front_only':
            selective_meta_blocks = [True] + [False] * (num_blocks - 1)

        # define head module
        self.head = common.default_conv(in_features, n_feats, kernel_size)

        # define body module
        if selective_meta_blocks is None:
            m_body = [
                ParamResBlock(
                    common.default_conv, n_feats, input_para, kernel_size, res_scale=res_scale,
                    q_layer_nonlinearity=q_layer_nonlinearity, add_q_layer=True, num_layers=num_layers
                ) for _ in range(n_resblocks)
            ]
        else:
            m_body = [
                ParamResBlock(
                    common.default_conv, n_feats, input_para, kernel_size, res_scale=res_scale,
                    q_layer_nonlinearity=q_layer_nonlinearity, add_q_layer=selective_meta_blocks[i],
                    num_layers=num_layers
                ) for i in range(n_resblocks)
            ]
        self.final_body = common.default_conv(n_feats, n_feats, kernel_size)

        # define tail module
        m_tail = [
            common.Upsampler(common.default_conv, scale, n_feats),
            common.default_conv(n_feats, out_features, kernel_size)
        ]

        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, metadata):
        x = self.head(x)
        res, _ = self.body((x, metadata))
        res = self.final_body(res)
        res += x
        x = self.tail(res)
        return x


class QSAN(nn.Module):
    """
    modified QSAN to allow insertion of meta-attention.  Refer to original SAN model for info on parameters.
    """

    def __init__(self, n_resgroups=20, n_resblocks=10, n_feats=64, reduction=16, scale=4, rgb_range=255, n_colors=3,
                 res_scale=1, conv=common.default_conv, input_para=1, num_q_blocks='all', num_inner_q_blocks=1,
                 **kwargs):
        super(QSAN, self).__init__()

        n_resgroups = n_resgroups
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        reduction = reduction
        scale = scale
        act = nn.ReLU(inplace=True)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        ##
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = 0.2
        self.n_resgroups = n_resgroups

        if num_q_blocks == 'all':
            self.RG = nn.ModuleList([QLSRAG(conv, n_feats, kernel_size, reduction, num_metadata=input_para,
                                            act=act, res_scale=res_scale, n_resblocks=n_resblocks) for _ in
                                     range(n_resgroups)])
        else:
            modules = []
            for i in range(n_resgroups):
                if i < num_q_blocks:
                    modules.append(QLSRAG(conv, n_feats, kernel_size, reduction, num_metadata=input_para,
                                          num_q_layers=num_inner_q_blocks, act=act, res_scale=res_scale,
                                          n_resblocks=n_resblocks))
                else:
                    modules.append(LSRAG(conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale,
                                         n_resblocks=n_resblocks))
            self.RG = nn.ModuleList(modules)

        self.conv_last = conv(n_feats, n_feats, kernel_size)

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        self.non_local = Nonlocal_CA(in_feat=n_feats, inter_feat=n_feats // 8, reduction=8, sub_sample=False,
                                     bn_layer=False)

        self.head = nn.Sequential(*modules_head)
        # self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)

        return nn.ModuleList(layers)

    def forward(self, x, metadata):

        x = self.head(x)

        # add nonlocal
        xx = self.non_local(x)

        # share-source skip connection
        residual = xx

        # share-source residual gruop
        for i, l in enumerate(self.RG):
            if isinstance(l, LSRAG):
                xx = l(xx) + self.gamma * residual
            else:
                xx = l((xx, metadata))[0] + self.gamma * residual

        # add nonlocal
        res = self.non_local(xx)
        res = res + x

        x = self.tail(res)

        return x


class QHAN(nn.Module):
    """
    Modified HAN network to include meta-attention.  Refer to original model for info on parameters.
    """

    def __init__(self, n_resgroups=10, n_resblocks=20, n_feats=64, reduction=16, num_metadata=0,
                 scale=4, n_colors=3, res_scale=1.0, style='standard', include_pixel_attention=False,
                 selective_meta_blocks=None,
                 include_q_layer=False, num_q_layers_inner_residual=None, num_layers_in_q_layer=2,
                 include_sft_layer=False, num_sft_layers_inner_residual=None,
                 include_dgfmb_layer=False, num_dgfmb_layers_inner_residual=None, num_layers_in_dgfmb_layer=2,
                 use_dgfmb_reduction=True, use_dgfmb_outer_reduction=False,
                 include_da_conv_layer=False, num_da_conv_layers_inner_residual=None, **kwargs):
        super(QHAN, self).__init__()

        n_resgroups = n_resgroups
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        reduction = reduction
        scale = scale
        act = nn.ReLU(True)

        # define head module
        modules_head = [common.default_conv(n_colors, n_feats, kernel_size)]

        # define body module
        if selective_meta_blocks is None:
            modules_body = [
                QResidualGroup(common.default_conv, n_feats, kernel_size, reduction, style=style,
                               num_metadata=num_metadata, pa=include_pixel_attention,
                               q_layer=include_q_layer,
                               dgfmb_layer=None,
                               sft_layer=include_sft_layer,
                               da_conv_layer=include_da_conv_layer,
                               act=act,
                               res_scale=res_scale,
                               n_resblocks=n_resblocks,
                               num_q_layers=num_q_layers_inner_residual, num_layers_in_q_layer=num_layers_in_q_layer,
                               num_dgfmb_layers=num_dgfmb_layers_inner_residual,
                               num_sft_layers=num_sft_layers_inner_residual,
                               num_layers_in_dgfmb_layer=num_layers_in_dgfmb_layer,
                               use_dgfmb_reduction=use_dgfmb_reduction,
                               num_da_conv_layers=num_da_conv_layers_inner_residual) for _ in range(n_resgroups)]
        else:
            modules_body = []

            for index in range(n_resgroups):
                if selective_meta_blocks[index]:
                    include_q = include_q_layer
                    include_dgfmb = include_dgfmb_layer
                    include_da_conv = include_da_conv_layer
                    include_sft = include_sft_layer
                else:
                    include_q = False
                    include_dgfmb = False
                    include_da_conv = False
                    include_sft = False

                modules_body.append(
                    QResidualGroup(common.default_conv, n_feats, kernel_size, reduction, style=style,
                                   num_metadata=num_metadata, pa=include_pixel_attention,
                                   q_layer=include_q, dgfmb_layer=include_dgfmb, da_conv_layer=include_da_conv,
                                   sft_layer=include_sft,
                                   act=act,
                                   res_scale=res_scale,
                                   n_resblocks=n_resblocks,
                                   num_q_layers=num_q_layers_inner_residual,
                                   num_layers_in_q_layer=num_layers_in_q_layer,
                                   num_dgfmb_layers=num_dgfmb_layers_inner_residual,
                                   num_layers_in_dgfmb_layer=num_layers_in_dgfmb_layer,
                                   num_sft_layers=num_sft_layers_inner_residual,
                                   use_dgfmb_reduction=use_dgfmb_reduction,
                                   num_da_conv_layers=num_da_conv_layers_inner_residual))

        modules_body.append(common.default_conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(common.default_conv, scale, n_feats, act=False),
            common.default_conv(n_feats, n_colors, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.csa = CSAM_Module(n_feats)
        self.la = LAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats * 11, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, metadata):

        x = self.head(x)
        res = x

        for name, midlayer in self.body._modules.items():
            if type(midlayer).__name__ == 'QResidualGroup':
                res, _ = midlayer((res, metadata))
            else:
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


def make_q_layer(basic_block, num_basic_block, num_feat,
                 q_block, num_q_blocks, num_q_blocks_inner, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for i in range(num_basic_block):
        if i < num_q_blocks:
            q_in = q_block
        else:
            q_in = None

        layers.append(basic_block(q_in, num_q_blocks_inner, num_feat, **kwarg))
    return nn.Sequential(*layers)


class QRRDB(nn.Module):
    """
    Meta-attention version of the Residual in Residual Dense Block.
    Used in RRDB-Net in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, q_block, num_q_blocks_inner, num_feat, num_grow_ch=32):
        super(QRRDB, self).__init__()

        self.q_block = q_block
        self.num_q_blocks_inner = num_q_blocks_inner

        if q_block:
            if q_block['q_block_name'] == 'q-layer':
                if num_q_blocks_inner > 0:
                    self.attention_block_1 = ParaCALayer(network_channels=q_block['network_channels'],
                                                         num_metadata=q_block['num_metadata'],
                                                         nonlinearity=q_block['nonlinearity'],
                                                         num_layers=q_block['num_layers'])
                if num_q_blocks_inner > 1:
                    self.attention_block_2 = ParaCALayer(network_channels=q_block['network_channels'],
                                                         num_metadata=q_block['num_metadata'],
                                                         nonlinearity=q_block['nonlinearity'],
                                                         num_layers=q_block['num_layers'])
                if num_q_blocks_inner > 2:
                    self.attention_block_3 = ParaCALayer(network_channels=q_block['network_channels'],
                                                         num_metadata=q_block['num_metadata'],
                                                         nonlinearity=q_block['nonlinearity'],
                                                         num_layers=q_block['num_layers'])
            elif q_block['q_block_name'] == 'da-layer':
                if num_q_blocks_inner > 0:
                    self.attention_block_1 = DA_conv(channels_in=q_block['channels_in'],
                                                     channels_out=q_block['channels_out'])
                if num_q_blocks_inner > 1:
                    self.attention_block_2 = DA_conv(channels_in=q_block['channels_in'],
                                                     channels_out=q_block['channels_out'])
                if num_q_blocks_inner > 2:
                    self.attention_block_3 = DA_conv(channels_in=q_block['channels_in'],
                                                     channels_out=q_block['channels_out'])
            elif q_block['q_block_name'] == 'dgfmb-layer':
                if num_q_blocks_inner > 0:
                    self.attention_block_1 = DGFMBLayer(num_channels=q_block['num_channels'],
                                                        degradation_full_dim=q_block['degradation_full_dim'],
                                                        num_layers=q_block['num_layers'],
                                                        use_linear=q_block['use_linear'])
                if num_q_blocks_inner > 1:
                    self.attention_block_2 = DGFMBLayer(num_channels=q_block['num_channels'],
                                                        degradation_full_dim=q_block['degradation_full_dim'],
                                                        num_layers=q_block['num_layers'],
                                                        use_linear=q_block['use_linear'])
                if num_q_blocks_inner > 2:
                    self.attention_block_3 = DGFMBLayer(num_channels=q_block['num_channels'],
                                                        degradation_full_dim=q_block['degradation_full_dim'],
                                                        num_layers=q_block['num_layers'],
                                                        use_linear=q_block['use_linear'])

        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        input_data = x
        if self.q_block:
            input_data = x[0]

        out = self.rdb1(input_data)

        if self.num_q_blocks_inner > 0 and self.q_block:
            out = self.attention_block_1(out, x[1])

        out = self.rdb2(out)
        if self.num_q_blocks_inner > 1 and self.q_block:
            out = self.attention_block_2(out, x[1])

        out = self.rdb3(out)
        if self.num_q_blocks_inner > 2 and self.q_block:
            out = self.attention_block_3(out, x[1])

        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + input_data


class QRRDBNet(nn.Module):
    """
    Meta-attention version of the RRDB network.
    Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32,
                 num_metadata=1, meta_block=None,
                 num_layers_in_q_block=2, num_q_blocks_inner_block=None,
                 num_q_blocks=None, **kwargs):
        super(QRRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        if meta_block == 'q-layer':
            q_block_in = {'q_block_name': meta_block,
                          'num_metadata': num_metadata,
                          'network_channels': num_feat,
                          'nonlinearity': True,
                          'num_layers': num_layers_in_q_block}
        elif meta_block == 'da-layer':
            q_block_in = {'q_block_name': meta_block,
                          'channels_in': num_metadata,
                          'channels_out': num_feat}
        elif meta_block == 'dgfmb-layer':
            q_block_in = {'q_block_name': meta_block,
                          'num_channels': num_feat,
                          'degradation_full_dim': num_metadata,
                          'num_layers': num_layers_in_q_block,
                          'use_linear': False}
        else:
            q_block_in = None

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_q_layer(basic_block=QRRDB, num_basic_block=num_block,
                                 q_block=q_block_in, num_q_blocks=num_q_blocks,
                                 num_q_blocks_inner=num_q_blocks_inner_block,
                                 num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, metadata):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body((feat, metadata)))
        feat = feat + body_feat
        # upsample

        # NOTE: This is a bit of a workaround for now to handle x8 SR
        # TODO: Check how to do it properly and allow more types of scale factors
        if self.scale == 8:
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=4, mode='nearest')))
        else:
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))

        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class QELAB(nn.Module):
    def __init__(self, inp_channels, out_channels, q_block,
                 exp_ratio=2, shifts=0,
                 window_sizes=[4, 8, 12], shared_depth=1, meta_placement='last_pass'):
        super(QELAB, self).__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.shared_depth = shared_depth
        self.meta_placement = meta_placement

        modules_lfe = {}
        modules_gmsa = {}
        modules_lfe['lfe_0'] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        modules_gmsa['gmsa_0'] = GMSA(channels=inp_channels, shifts=shifts, window_sizes=window_sizes, calc_attn=True)
        for i in range(shared_depth):
            modules_lfe['lfe_{}'.format(i + 1)] = LFE(inp_channels=inp_channels,
                                                      out_channels=out_channels, exp_ratio=exp_ratio)
            modules_gmsa['gmsa_{}'.format(i + 1)] = GMSA(channels=inp_channels, shifts=shifts,
                                                         window_sizes=window_sizes, calc_attn=False)
        self.modules_lfe = nn.ModuleDict(modules_lfe)
        self.modules_gmsa = nn.ModuleDict(modules_gmsa)

        if q_block is not None:
            if q_block['q_block_name'] == 'q-layer':
                self.meta_attention = ParaCALayer(network_channels=q_block['network_channels'],
                                                  num_metadata=q_block['num_metadata'],
                                                  nonlinearity=q_block['nonlinearity'],
                                                  num_layers=q_block['num_layers'])
            elif q_block['q_block_name'] == 'da-layer':
                self.meta_attention = DA_conv(channels_in=q_block['channels_in'],
                                              channels_out=q_block['channels_out'])
            elif q_block['q_block_name'] == 'dgfmb-layer':
                self.meta_attention = DGFMBLayer(num_channels=q_block['num_channels'],
                                                 degradation_full_dim=q_block['degradation_full_dim'],
                                                 num_layers=q_block['num_layers'],
                                                 use_linear=q_block['use_linear'])
        else:
            self.meta_attention = None

    def forward(self, x):
        xf, metadata = x[0], x[1]

        if self.meta_attention and self.meta_placement == 'first_pass':
            xf = self.meta_attention(xf, metadata)

        atn = None
        for i in range(1 + self.shared_depth):
            if i == 0:  # only calculate attention for the 1-st module
                xf = self.modules_lfe['lfe_{}'.format(i)](xf) + xf
                y, atn = self.modules_gmsa['gmsa_{}'.format(i)](xf, None)
                xf = y + xf
            else:
                xf = self.modules_lfe['lfe_{}'.format(i)](xf) + xf
                y, atn = self.modules_gmsa['gmsa_{}'.format(i)](xf, atn)
                xf = y + xf

        if self.meta_attention and self.meta_placement == 'last_pass':
            xf = self.meta_attention(xf, metadata)
        return xf, metadata


class QELAN(nn.Module):
    """
    Based on the implementation from: https://github.com/xindongzhang/ELAN/blob/main/models/elan_network.py
    """

    def __init__(self, scale=4,
                 colors=3, window_sizes=[4, 8, 16],
                 m_elan=36, c_elan=180,
                 n_share=0, r_expand=2, rgb_range=1.0,
                 num_metadata=1, meta_block=None,
                 num_layers_in_q_block=2,
                 apply_mean_shift=True,
                 meta_placement='last_pass',
                 num_q_blocks=None, **kwargs):
        super(QELAN, self).__init__()

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

        if meta_block == 'q-layer':
            q_block_in = {'q_block_name': meta_block,
                          'num_metadata': num_metadata,
                          'network_channels': c_elan,
                          'nonlinearity': True,
                          'num_layers': num_layers_in_q_block}
        elif meta_block == 'da-layer':
            q_block_in = {'q_block_name': meta_block,
                          'channels_in': num_metadata,
                          'channels_out': c_elan}
        elif meta_block == 'dgfmb-layer':
            q_block_in = {'q_block_name': meta_block,
                          'num_channels': c_elan,
                          'degradation_full_dim': num_metadata,
                          'num_layers': num_layers_in_q_block,
                          'use_linear': False}
        else:
            q_block_in = None

        # define head module
        m_head = [nn.Conv2d(self.colors, self.c_elan, kernel_size=3, stride=1, padding=1)]

        # define body module
        m_body = []
        for i in range(self.m_elan // (1 + self.n_share)):
            if i >= num_q_blocks:
                q_block_in = None

            if (i + 1) % 2 == 1:
                m_body.append(
                    QELAB(
                        self.c_elan, self.c_elan, q_block_in, self.r_expand, 0,
                        self.window_sizes, shared_depth=self.n_share, meta_placement=meta_placement
                    )
                )
            else:
                m_body.append(
                    QELAB(
                        self.c_elan, self.c_elan, q_block_in, self.r_expand, 1,
                        self.window_sizes, shared_depth=self.n_share, meta_placement=meta_placement
                    )
                )
        # define tail module
        m_tail = [
            nn.Conv2d(self.c_elan, self.colors * self.scale * self.scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, metadata):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        if self.apply_mean_shift:
            x = self.sub_mean(x)
        x = self.head(x)
        res, _ = self.body((x, metadata))
        res = res + x
        x = self.tail(res)
        if self.apply_mean_shift:
            x = self.add_mean(x)

        return x[:, :, 0:H * self.scale, 0:W * self.scale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize * self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
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
