# Code modified from https://github.com/Maclory/Deep-Iterative-Collaboration
import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, FeatureHeatmapFusingBlock
from .modules.architecture import FeedbackHourGlass
from .srfbn_hg_arch import FeedbackBlockCustom, FeedbackBlockHeatmapAttention, merge_heatmap_5


class DIC(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        in_channels = kwargs['in_channels']
        out_channels = kwargs['out_channels']
        num_groups = kwargs['num_groups']
        hg_num_feature = kwargs['hg_num_feature']
        hg_num_keypoints = kwargs['hg_num_keypoints']
        act_type = 'prelu'
        norm_type = None

        self.num_steps = kwargs['num_steps']
        num_features = kwargs['num_features']
        self.upscale_factor = kwargs['scale']
        self.detach_attention = kwargs['detach_attention']
        if self.detach_attention:
            print('Attention detached')
        else:
            print('Attention attached')

        if self.upscale_factor == 8:
            # with PixelShuffle at start, need to upscale 4x only
            stride = 4
            padding = 2
            kernel_size = 8
        elif self.upscale_factor == 4:
            stride = 2
            padding = 1
            kernel_size = 4
        else:
            raise NotImplementedError("Upscale factor %d not implemented!" % self.upscale_factor)

        # LR feature extraction block
        self.conv_in = ConvBlock(
            in_channels,
            4 * num_features,
            kernel_size=3,
            act_type=act_type,
            norm_type=norm_type)
        self.feat_in = nn.PixelShuffle(2)

        # basic block
        self.first_block = FeedbackBlockCustom(num_features, num_groups, self.upscale_factor,
                                               act_type, norm_type, num_features)
        self.block = FeedbackBlockHeatmapAttention(num_features, num_groups, self.upscale_factor, act_type, norm_type, 5, kwargs['num_fusion_block'], device=device)
        self.block.should_reset = False

        # reconstruction block
        # self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')
        #TODO: modification point here - check what needs to be done to get a X2 upscailing
        self.out = DeconvBlock(
            num_features,
            num_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act_type='prelu',
            norm_type=norm_type)
        self.conv_out = ConvBlock(
            num_features,
            out_channels,
            kernel_size=3,
            act_type=None,
            norm_type=norm_type)

        self.HG = FeedbackHourGlass(hg_num_feature, hg_num_keypoints, self.upscale_factor)

    def forward(self, x):
        inter_res = nn.functional.interpolate(
            x,
            scale_factor=self.upscale_factor,
            mode='bilinear',
            align_corners=False)

        batch_size = x.size(0)

        x = self.conv_in(x)
        x = self.feat_in(x)
        sr_outs = []
        heatmap_outs = []
        hg_last_hidden = None

        # initalize heatmap and FB feature with first coarse block

        for step in range(self.num_steps):
            if step == 0:
                FB_out_first = self.first_block(x)
                h = torch.add(inter_res, self.conv_out(self.out(FB_out_first)))
                heatmap, hg_last_hidden = self.HG(h, hg_last_hidden)
                self.block.last_hidden = FB_out_first
                assert self.block.should_reset == False
            else:
                FB_out = self.block(x, merge_heatmap_5(heatmap, self.detach_attention))
                h = torch.add(inter_res, self.conv_out(self.out(FB_out)))
                heatmap, hg_last_hidden = self.HG(h, hg_last_hidden)

            sr_outs.append(h)
            heatmap_outs.append(heatmap)

        return sr_outs, heatmap_outs  # return output of every timesteps
