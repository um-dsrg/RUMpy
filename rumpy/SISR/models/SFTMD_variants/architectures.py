import torch
import torch.nn as nn
import torch.nn.functional as F
from rumpy.SISR.models.attention_manipulators.q_layer import ParaCALayer
from rumpy.SISR.models.attention_manipulators.da_layer import DA_conv


# original SFTMD code based on https://github.com/yuanjunchai/IKC/
class ConcatSft(nn.Module):
    def __init__(self, nf=64, para=1, **kwargs):
        super(ConcatSft, self).__init__()
        self.conv = nn.Conv2d(para + nf, nf, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps, para_maps):
        return self.conv(torch.cat((feature_maps, para_maps), dim=1))


class WeakSft(nn.Module):
    def __init__(self):
        super(WeakSft, self).__init__()

    def forward(self, feature_maps, para_maps):
        return feature_maps * para_maps


class StandardSft(nn.Module):
    def __init__(self, nf=64, para=1, mask_para=False, repeats=None, **kwargs):
        super(StandardSft, self).__init__()
        self.mask_para = mask_para
        self.repeats = repeats

        if mask_para:
            para = 0

        if repeats is not None:
            para = para * repeats

        self.mul_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

        self.add_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps, para_maps):
        if self.repeats is not None:
            para_maps = para_maps.repeat(1, self.repeats, 1, 1)
        if self.mask_para:
            cat_input = feature_maps
        else:
            cat_input = torch.cat((feature_maps, para_maps), dim=1)

        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))

        return feature_maps * mul + add


class SplitSft(nn.Module):
    def __init__(self, nf=64, para=1, mask_para=False, split='22', **kwargs):
        super(SplitSft, self).__init__()

        self.mask_para = mask_para
        if mask_para:
            para = 0

        conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        relu_conv1 = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(para + nf, nf, kernel_size=3, stride=1, padding=1)
        relu_conv2 = nn.ReLU(inplace=True)
        conv3 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        relu_conv3 = nn.ReLU(inplace=True)
        conv4 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        relu_conv4 = nn.ReLU(inplace=True)

        if split == '13':
            self.f_path = nn.Sequential(conv1, relu_conv1)
            self.q_path = nn.Sequential(conv2, relu_conv2, conv3, relu_conv3, conv4, relu_conv4)
        elif split == '22':
            self.f_path = nn.Sequential(conv1, relu_conv1, conv4, relu_conv4)
            self.q_path = nn.Sequential(conv2, relu_conv2, conv3, relu_conv3)
        elif split == '31':
            self.f_path = nn.Sequential(conv1, relu_conv1, conv3, relu_conv3, conv4, relu_conv4)
            self.q_path = nn.Sequential(conv2, relu_conv2)
        elif split == '04':
            self.f_path = nn.Sequential()
            self.q_path = nn.Sequential(conv2, relu_conv2, conv1, relu_conv1, conv3, relu_conv3, conv4, relu_conv4)
        elif split == '40':
            conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
            self.f_path = nn.Sequential(conv1, relu_conv1, conv2, relu_conv2, conv3, relu_conv3, conv4, relu_conv4)
            self.q_path = nn.Sequential()

    def forward(self, feature_maps, para_maps):

        if self.mask_para:
            cat_input = feature_maps
        else:
            cat_input = torch.cat((feature_maps, para_maps), dim=1)

        f_path = self.f_path(feature_maps)
        q_path = self.q_path(cat_input)

        return f_path + q_path


class SFT_Layer(nn.Module):
    def __init__(self, sft_type='standard', **kwargs):
        super(SFT_Layer, self).__init__()

        if sft_type == 'weak':
            self.sft_module = WeakSft()
        elif sft_type == 'concat':
            self.sft_module = ConcatSft(**kwargs)
        elif sft_type == 'standard':
            self.sft_module = StandardSft(**kwargs)
        elif sft_type == 'split':
            self.sft_module = SplitSft(**kwargs)
        elif sft_type == 'none':
            self.sft_module = None

    def forward(self, feature_maps, para_maps):
        if self.sft_module is None:
            return feature_maps
        else:
            return self.sft_module(feature_maps, para_maps)


class SFT_Residual_Block(nn.Module):
    def __init__(self, nf=64, para=1, SFT_type='standard', mask_para=False, repeats=None, q_injection=False,
                 da_injection=False, q_layers=2, split='22'):
        super(SFT_Residual_Block, self).__init__()

        self.sft1 = SFT_Layer(nf=nf, para=para, mask_para=mask_para, repeats=repeats, sft_type=SFT_type, split=split)
        self.sft2 = SFT_Layer(nf=nf, para=para, mask_para=mask_para, repeats=repeats, sft_type=SFT_type, split=split)

        self.conv1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True)

        self.q_injection = q_injection
        self.da_injection = da_injection
        if q_injection:
            self.q_1 = ParaCALayer(network_channels=nf, num_metadata=para, nonlinearity=True, num_layers=q_layers)
            self.q_2 = ParaCALayer(network_channels=nf, num_metadata=para, nonlinearity=True, num_layers=q_layers)
        if da_injection:
            self.d_1 = DA_conv(channels_in=para, channels_out=nf)
            self.d_2 = DA_conv(channels_in=para, channels_out=nf)

    def forward(self, feature_maps, para_maps):

        fea1 = F.relu(self.sft1(feature_maps, para_maps))
        if self.q_injection:
            fea1 = self.q_1(fea1, para_maps)
        if self.da_injection:
            fea1 = self.d_1((fea1, para_maps))
        fea2 = F.relu(self.sft2(self.conv1(fea1), para_maps))
        if self.q_injection:
            fea2 = self.q_2(fea2, para_maps)
        if self.da_injection:
            fea2 = self.d_2((fea2, para_maps))
        fea3 = self.conv2(fea2)
        return torch.add(feature_maps, fea3)


class SFTMD(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, num_features=64, num_blocks=16, scale=4, input_para=1, split='22',
                 SFT_type='standard', mask_para=False, repeats=None, q_injection=False, da_injection=False,
                 q_layers=2, **kwargs):
        super(SFTMD, self).__init__()
        self.min = 0.0
        self.max = 1.0
        self.para = input_para
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv2d(in_nc, num_features, 3, stride=1, padding=1)
        self.relu_conv1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, stride=1, padding=1)
        self.relu_conv2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(num_features, num_features, 3, stride=1, padding=1)

        for i in range(num_blocks):
            self.add_module('SFT-residual' + str(i + 1), SFT_Residual_Block(nf=num_features, para=input_para,
                                                                            SFT_type=SFT_type, split=split,
                                                                            da_injection=da_injection,
                                                                            q_injection=q_injection, q_layers=q_layers,
                                                                            mask_para=mask_para, repeats=repeats))

        self.sft = SFT_Layer(nf=num_features, para=input_para, mask_para=mask_para, repeats=repeats,
                             split=split, sft_type=SFT_type)

        self.q_injection = q_injection
        self.da_injection = da_injection

        if q_injection:
            self.final_injection = ParaCALayer(network_channels=num_features, num_metadata=input_para,
                                               nonlinearity=True, num_layers=q_layers)
        elif da_injection:
            self.final_injection = DA_conv(channels_in=input_para, channels_out=num_features)

        self.conv_mid = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1,
                                  padding=1, bias=True)

        if scale == 4:  # x4
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=num_features, out_channels=num_features * scale, kernel_size=3, stride=1,
                          padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=num_features, out_channels=num_features * scale, kernel_size=3, stride=1,
                          padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:  # x2, x3
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=num_features, out_channels=num_features * scale ** 2, kernel_size=3, stride=1,
                          padding=1, bias=True),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_output = nn.Conv2d(in_channels=num_features, out_channels=out_nc, kernel_size=9, stride=1, padding=4,
                                     bias=True)

    def forward(self, x, metadata):
        fea_bef = self.conv3(self.relu_conv2(self.conv2(self.relu_conv1(self.conv1(x)))))
        fea_in = fea_bef
        for i in range(self.num_blocks):
            fea_in = self.__getattr__('SFT-residual' + str(i + 1))(fea_in, metadata)
        fea_mid = fea_in
        fea_add = torch.add(fea_mid, fea_bef)
        fea_fin = self.sft(fea_add, metadata)
        if self.q_injection:
            fea_fin = self.final_injection(fea_fin, metadata)
        elif self.da_injection:
            fea_fin = self.final_injection((fea_fin, metadata))
        fea = self.upscale(self.conv_mid(fea_fin))
        out = self.conv_output(fea)

        return torch.clamp(out, min=self.min, max=self.max)



