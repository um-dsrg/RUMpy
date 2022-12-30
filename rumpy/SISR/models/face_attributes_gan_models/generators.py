import torch
import torch.nn.functional as F
from torch import nn

from rumpy.SISR.models.face_attributes_gan_models.common_blocks import *


class STN_L1_UpG(nn.Module):
    """
    Spatial Transformer Network L1
    Based on implementation in https://github.com/XinYuANU/FaceAttr/blob/master/stn_L1_UpG.lua
    """

    def __init__(self):
        super(STN_L1_UpG, self).__init__()

        # Localisation network
        self.locnet_body = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 20, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 20, 3),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(20*2*2, 20),
            nn.ReLU(inplace=True)
        )

        # This is the one in the GitHub
        # self.locnet_out_layer = nn.Linear(20, 4)
        # self.locnet_out_layer.weight.data.zero_()
        # self.locnet_out_layer.bias.data.copy_(torch.tensor([0, 1, 1, 0], dtype=torch.float))

        # But this is what PyTorch expects
        self.locnet_out_layer = nn.Linear(20, 6)
        self.locnet_out_layer.weight.data.zero_()
        identity_tensor = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        self.locnet_out_layer.bias.data.copy_(identity_tensor)

    # Following this example: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    def forward(self, x):
        xs = self.locnet_body(x)

        theta = self.locnet_out_layer(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, [x.size(dim=0), x.size(dim=1), 32, 32], align_corners=True)
        out = F.grid_sample(x, grid, align_corners=True)

        return out


class STN_L2_UpG(nn.Module):
    """
    Spatial Transformer Network L1
    Based on implementation in https://github.com/XinYuANU/FaceAttr/blob/master/stn_L2_UpG.lua
    """

    def __init__(self):
        super(STN_L2_UpG, self).__init__()

        # Localisation network
        self.locnet_body = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 20, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 20, 3),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(20*3*3, 20),
            nn.ReLU(inplace=True)
        )

        # This is the one in the GitHub
        # self.locnet_out_layer = nn.Linear(20, 4)
        # self.locnet_out_layer.weight.data.zero_()
        # self.locnet_out_layer.bias.data.copy_(torch.tensor([0, 1, 1, 0], dtype=torch.float))

        # But this is what PyTorch expects
        self.locnet_out_layer = nn.Linear(20, 6)
        self.locnet_out_layer.weight.data.zero_()
        identity_tensor = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        self.locnet_out_layer.bias.data.copy_(identity_tensor)

    # Following this example: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    def forward(self, x):
        xs = self.locnet_body(x)

        theta = self.locnet_out_layer(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, [x.size(dim=0), x.size(dim=1), 64, 64], align_corners=True)
        out = F.grid_sample(x, grid, align_corners=True)

        return out


class FaceSRAttributesGeneratorNet(nn.Module):
    """
    Based on implementation in https://github.com/XinYuANU/FaceAttr/blob/master/train_encoder_decoder_Stack_skip_perception.lua
    """

    def __init__(self, n_feats=32, n_attributes=18, remove_stn=False, use_attribute_encoder=False):
        super(FaceSRAttributesGeneratorNet, self).__init__()
        self.n_attributes = n_attributes

        # Explanation of the '-' symbol here: https://github.com/torch/nngraph#a-network-with-2-inputs-and-2-outputs
        # The minus sign is a more compact way of showing the flow of the tensors through the model.
        # After finally figuring out this notation together with some other Lua notation,
        # it should be easy to replicate the models.

        self.generator_encoder_1 = nn.Sequential(
            nn.Conv2d(3, n_feats, 4, 2, 1),
            nn.BatchNorm2d(num_features=n_feats),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.generator_encoder_2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats*4, 4, 2, 1),
            nn.BatchNorm2d(num_features=n_feats*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.generator_encoder_3 = nn.Sequential(
            nn.Conv2d(n_feats*4, n_feats*16, 4, 2, 1),
            nn.BatchNorm2d(num_features=n_feats*16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.generator_encoder_4 = nn.Sequential(
            nn.Conv2d(n_feats*16, n_feats*64, 2, 1),
            nn.BatchNorm2d(num_features=n_feats*64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.generator_decoder_1 = nn.Sequential(
            nn.ConvTranspose2d((n_feats*64)+n_attributes, n_feats*32, 4, 2, 1),
            nn.BatchNorm2d(num_features=n_feats*32),
            nn.ReLU(inplace=True)
        )

        self.generator_decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(n_feats*48, n_feats*24, 4, 2, 1),
            nn.BatchNorm2d(num_features=n_feats*24),
            nn.ReLU(inplace=True)
        )

        self.generator_decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(n_feats*28, n_feats*16, 4, 2, 1),
            nn.BatchNorm2d(num_features=n_feats*16),
            nn.ReLU(inplace=True)
        )

        self.generator_decoder_4 = nn.Sequential(
            nn.ConvTranspose2d(n_feats*17, n_feats*8, 4, 2, 1),
            nn.BatchNorm2d(num_features=n_feats*8),
            nn.ReLU(inplace=True)
        )

        self.upsample_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            STN_L1_UpG(),
            nn.Conv2d(n_feats*8, n_feats*4, 3, 1, 1),
            nn.BatchNorm2d(num_features=n_feats*4),
            nn.ReLU(inplace=True)
        )

        self.upsample_2 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            STN_L2_UpG(),
            nn.Conv2d(n_feats*4, n_feats*2, 3, 1, 1),
            nn.BatchNorm2d(num_features=n_feats*2),
            nn.ReLU(inplace=True)
        )

        if remove_stn:
            self.upsample_1 = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(n_feats*8, n_feats*4, 3, 1, 1),
                nn.BatchNorm2d(num_features=n_feats*4),
                nn.ReLU(inplace=True)
            )

            self.upsample_2 = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(n_feats*4, n_feats*2, 3, 1, 1),
                nn.BatchNorm2d(num_features=n_feats*2),
                nn.ReLU(inplace=True)
            )

        self.upsample_final_layer = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(n_feats*2, n_feats, 3, 1, 1),
            nn.BatchNorm2d(num_features=n_feats),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, 3, 5, 1, 2)
        )

        # Adding this to see if perhaps modifying the attributes before concatenation will help
        self.metadata_layers = nn.Sequential(nn.Identity())

        if use_attribute_encoder:
            self.metadata_layers = nn.Sequential(
                nn.Conv2d(n_attributes, n_attributes*2, 1, 1),
                nn.Conv2d(n_attributes*2, n_attributes, 1, 1)
            )

    def forward(self, x, metadata):
        e_1 = self.generator_encoder_1(x)
        e_2 = self.generator_encoder_2(e_1)
        e_3 = self.generator_encoder_3(e_2)
        e_4 = self.generator_encoder_4(e_3)

        metadata_out = self.metadata_layers(metadata)

        e_5 = torch.cat((e_4, metadata_out), dim=1)

        d_1 = self.generator_decoder_1(e_5)
        d_1_1 = torch.cat((d_1, e_3), dim=1)
        d_2 = self.generator_decoder_2(d_1_1)
        d_2_1 = torch.cat((d_2, e_2), dim=1)
        d_3 = self.generator_decoder_3(d_2_1)
        d_3_1 = torch.cat([d_3, e_1], dim=1)
        d_4 = self.generator_decoder_4(d_3_1)

        u_1 = self.upsample_1(d_4)
        u_2 = self.upsample_2(u_1)
        out = self.upsample_final_layer(u_2)

        return out


class ConvPixelShuffleReLU(nn.Module):
    """
    Based on implementation in https://github.com/NoviceMAn-prog/AGA-GAN/blob/main/agagan_unet.py
    """

    def __init__(self, in_feats=128, out_feats=128, scale=2):
        super(ConvPixelShuffleReLU, self).__init__()

        self.conv_shuffle_relu = nn.Sequential(
            nn.Conv2d(in_feats, out_feats*scale*scale, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv_shuffle_relu(x)

        return out


class RDDB(nn.Module):
    """
    Based on implementation in https://github.com/NoviceMAn-prog/AGA-GAN/blob/main/agagan_unet.py
    """

    def __init__(self, in_feats=128, out_feats=128, n_feats=64):
        super(RDDB, self).__init__()

        self.rddb_conv_lrelu_head = nn.Sequential(
            nn.Conv2d(in_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.25),
        )

        self.rddb_conv_lrelu_body_1 = nn.Sequential(
            nn.Conv2d(in_feats+n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.25),
        )

        self.rddb_conv_lrelu_body_2 = nn.Sequential(
            nn.Conv2d(in_feats+n_feats*2, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.25),
        )

        self.rddb_conv_lrelu_body_3 = nn.Sequential(
            nn.Conv2d(in_feats+n_feats*3, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.25),
        )

        self.rddb_conv_lrelu_tail = nn.Sequential(
            nn.Conv2d(in_feats+n_feats*4, out_feats, 3, 1, 1),
            nn.LeakyReLU(0.25),
        )

    def forward(self, x):
        r1 = self.rddb_conv_lrelu_head(x)
        r1_c = torch.cat((x, r1), dim=1)

        r2 = self.rddb_conv_lrelu_body_1(r1_c)
        r2_c = torch.cat((x, r1, r2), dim=1)

        r3 = self.rddb_conv_lrelu_body_2(r2_c)
        r3_c = torch.cat((x, r1, r2, r3), dim=1)

        r4 = self.rddb_conv_lrelu_body_3(r3_c)
        r4_c = torch.cat((x, r1, r2, r3, r4), dim=1)

        r5 = self.rddb_conv_lrelu_tail(r4_c)
        r5_l = r5 * 0.4

        out = r5_l + x

        return out


class SEBlock(nn.Module):
    def __init__(self, in_feats, ratio=16):
        super(SEBlock, self).__init__()

        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(in_feats, in_feats // ratio),
            nn.ReLU(),
            nn.Linear(in_feats // ratio, in_feats),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.se_block(x)
        out = att * x

        return out


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_feats, n_feats):
        super(SpatialAttentionBlock, self).__init__()

        self.sa_block = nn.Sequential(
            nn.Conv2d(in_feats, n_feats, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_feats, 1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.sa_block(x)

        return out


class DualAttentionBlock(nn.Module):
    def __init__(self, in_feats, skip_out_feats, out_feats):
        super(DualAttentionBlock, self).__init__()

        self.da_block_head = nn.Sequential(
            ConvPixelShuffleReLU(in_feats, out_feats),
            nn.ReLU()
        )

        self.shallow_conv = nn.Sequential(
            nn.Conv2d(skip_out_feats+out_feats, out_feats, 3, 1, 1),
            nn.ReLU()
        )

        self.se_block = nn.Sequential(
            SEBlock(out_feats)
        )

        self.sa_block = nn.Sequential(
            SpatialAttentionBlock(out_feats, out_feats // 4)
        )

    def forward(self, x, skip):
        up = self.da_block_head(x)
        up_c = torch.cat((skip, up), dim=1)

        conv = self.shallow_conv(up_c)

        se_block = self.se_block(conv)

        sa_block = self.sa_block(conv)
        sa_block = sa_block + 1

        out = se_block * sa_block

        return out


class AGAGANUNet(nn.Module):
    """
    Based on implementation in https://github.com/NoviceMAn-prog/AGA-GAN/blob/main/agagan_unet.py
    """

    def __init__(self, n_feats=32):
        super(AGAGANUNet, self).__init__()

        self.max_pool = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )

        self.unet_enc_head = nn.Sequential(
            nn.Conv2d(6, n_feats, 3, 1, 1),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.25)
        )

        self.se_block_enc_head = nn.Sequential(
            SEBlock(n_feats)
        )

        self.unet_enc_1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats*2, 3, 1, 1),
            nn.Conv2d(n_feats*2, n_feats*2, 3, 1, 1),
            nn.LeakyReLU(0.25)
        )

        self.se_block_enc_1 = nn.Sequential(
            SEBlock(n_feats*2)
        )

        self.unet_enc_2 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*4, 3, 1, 1),
            nn.Conv2d(n_feats*4, n_feats*4, 3, 1, 1),
            nn.LeakyReLU(0.25)
        )

        self.se_block_enc_2 = nn.Sequential(
            SEBlock(n_feats*4)
        )

        self.unet_enc_3 = nn.Sequential(
            nn.Conv2d(n_feats*4, n_feats*8, 3, 1, 1),
            nn.Conv2d(n_feats*8, n_feats*8, 3, 1, 1),
            nn.Conv2d(n_feats*8, n_feats*8, 3, 1, 1),
            nn.LeakyReLU(0.25)
        )

        self.se_block_enc_3 = nn.Sequential(
            SEBlock(n_feats*8)
        )

        self.da_block_dec_1 = nn.Sequential(
            DualAttentionBlock(n_feats*8, n_feats*4, n_feats*4)
        )

        self.unet_dec_1_1 = nn.Sequential(
            nn.Conv2d(n_feats*4, n_feats*4, 3, 1, 1)
        )

        self.unet_dec_1_2 = nn.Sequential(
            nn.Conv2d(n_feats*4, n_feats*4, 3, 1, 1),
            nn.LeakyReLU(0.25)
        )

        self.unet_dec_1_3 = nn.Sequential(
            nn.Conv2d(n_feats*4, n_feats*4, 3, 1, 1)
        )

        self.da_block_dec_2 = nn.Sequential(
            DualAttentionBlock(n_feats*4, n_feats*2, n_feats*2)
        )

        self.unet_dec_2_1 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, 3, 1, 1)
        )

        self.unet_dec_2_2 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, 3, 1, 1),
            nn.LeakyReLU(0.25)
        )

        self.da_block_dec_3 = nn.Sequential(
            DualAttentionBlock(n_feats*2, n_feats, n_feats)
        )

        self.unet_dec_3_1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

        self.unet_dec_3_2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.25)
        )

        self.unet_tail = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.unet_enc_head(x)
        x1_se = self.se_block_enc_head(x1)
        x1_mp = self.max_pool(x1_se)

        x2 = self.unet_enc_1(x1_mp)
        x2_se = self.se_block_enc_1(x2)
        x2_mp = self.max_pool(x2_se)

        x3 = self.unet_enc_2(x2_mp)
        x3_se = self.se_block_enc_2(x3)
        x3_mp = self.max_pool(x3_se)

        x4 = self.unet_enc_3(x3_mp)
        x4_se = self.se_block_enc_3(x4)

        x5 = self.da_block_dec_1(x4_se, x3_se)
        x5_1 = self.unet_dec_1_1(x5)
        x5_2 = self.unet_dec_1_2(x5_1)
        x5_a = x5_1 + x5_2
        x5_c = self.unet_dec_1_3(x5_a)

        x6 = self.da_block_dec_2(x5_c, x2_se)
        x6_1 = self.unet_dec_2_1(x6)
        x6_2 = self.unet_dec_2_2(x6_1)
        x6_a = x6_1 + x6_2

        x7 = self.da_block_dec_3(x6_a, x1_se)
        x7_1 = self.unet_dec_3_1(x7)
        x7_2 = self.unet_dec_3_2(x7_1)
        x7_a = x7_1 + x7_2

        out = self.unet_tail(x7_a)

        return out


class AGAGANGenerator(nn.Module):
    """
    Based on implementation in https://github.com/NoviceMAn-prog/AGA-GAN/blob/main/agagan_unet.py
    """

    def __init__(self, n_feats=32, n_attributes=38, use_transpose=True):
        super(AGAGANGenerator, self).__init__()

        # Attributes and Attention Blocks
        self.attributes_dense_block = nn.Sequential(
            nn.Linear(n_attributes, 768),
            nn.LeakyReLU(0.25)
        )

        self.shallow_conv_block = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.25),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.25),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

        self.shallow_conv_f1 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, 3, 1, 1),
            nn.LeakyReLU(0.25)
        )

        self.shallow_conv_f2 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*4, 3, 1, 1),
            nn.LeakyReLU(0.25)
        )

        self.shallow_conv_f3 = nn.Sequential(
            nn.Conv2d(n_feats*4, n_feats*4, 3, 1, 1),
            nn.LeakyReLU(0.25)
        )

        self.upsample_256_128_lrelu = nn.Sequential(
            ConvPixelShuffleReLU(n_feats*8, n_feats*4),
            nn.LeakyReLU(0.25)
        )

        self.upsample_192_64_lrelu = nn.Sequential(
            ConvPixelShuffleReLU(n_feats*6, n_feats*2),
            nn.LeakyReLU(0.25)
        )

        self.upsample_128_128_lrelu = nn.Sequential(
            ConvPixelShuffleReLU(n_feats*4, n_feats*4),
            nn.LeakyReLU(0.25)
        )

        self.upsample_128_64_lrelu = nn.Sequential(
            ConvPixelShuffleReLU(n_feats*4, n_feats*2),
            nn.LeakyReLU(0.25)
        )

        self.upsample_64_64_lrelu = nn.Sequential(
            ConvPixelShuffleReLU(n_feats*2, n_feats*2),
            nn.LeakyReLU(0.25)
        )

        # Basic Blocks
        self.upsample_only_wide = nn.Sequential(
            ConvPixelShuffleReLU(n_feats*4, n_feats*4)
        )

        self.upsample_only_narrow = nn.Sequential(
            ConvPixelShuffleReLU(n_feats*2, n_feats*2)
        )

        self.conv2d_transpose = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1)
        )

        if use_transpose:
            self.conv2d_transpose = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, 1, 1)
            )

        self.lrelu_only = nn.Sequential(
            nn.LeakyReLU(0.25)
        )

        self.conv_only_1 = nn.Sequential(
            nn.Conv2d(n_feats*4, n_feats*4, 3, 1, 1)
        )

        self.conv_only_2 = nn.Sequential(
            nn.Conv2d(n_feats*8, n_feats*4, 3, 1, 1)
        )

        self.conv_bottleneck_sigmoid = nn.Sequential(
            nn.Conv2d(n_feats*4, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        # Generator Blocks
        self.main_branch_head = nn.Sequential(
            nn.Conv2d(3, n_feats*2, 3, 1, 1),
            nn.LeakyReLU(0.25)
        )

        self.main_branch_body_1 = nn.Sequential(
            nn.Conv2d(n_feats*4, n_feats*4, 3, 1, 1),
            nn.LeakyReLU(0.25)
        )

        self.main_branch_body_2 = nn.Sequential(
            nn.Conv2d(n_feats*8, n_feats*4, 3, 1, 1),
            nn.LeakyReLU(0.25)
        )

        self.main_branch_body_3 = nn.Sequential(
            nn.Conv2d(n_feats*4, n_feats*4, 3, 1, 1),
            nn.LeakyReLU(0.25)
        )

        self.rddb_block = nn.Sequential(
            RDDB()
        )

        self.main_branch_tail = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x, metadata):
        lr = self.shallow_conv_block(x)

        # Convert from B x C x 1 x 1 to B x C for Dense/Linear layer
        metadata_r = torch.squeeze(metadata)
        att = self.attributes_dense_block(metadata_r)
        att_r = torch.reshape(att, (-1, 3, 16, 16))
        att_f = self.shallow_conv_block(att_r)

        lr_att_combined = torch.cat((att_f, lr), dim=1)

        f1 = self.shallow_conv_f1(lr_att_combined)
        f2 = self.shallow_conv_f2(f1)
        f3 = self.shallow_conv_f3(f2)
        f4 = self.upsample_128_128_lrelu(f3)

        conv1 = self.main_branch_head(x)
        conv1_c = torch.cat((conv1, f1), dim=1)

        conv2 = self.main_branch_body_1(conv1_c)

        rddb1 = self.rddb_block(conv2)
        rddb1_c = torch.cat((rddb1, f2), dim=1)

        conv3 = self.main_branch_body_2(rddb1_c)

        rddb2 = self.rddb_block(conv3)
        rddb2_c = torch.cat((rddb2, f3), dim=1)

        conv4 = self.main_branch_body_2(rddb2_c)

        rddb3 = self.rddb_block(conv4)
        rddb3_l = rddb3 * 0.4

        rddb_out = rddb3_l + conv2

        conv5 = self.main_branch_body_3(rddb_out)

        up_conv4 = self.upsample_only_wide(conv5)

        up_conv4_without = self.lrelu_only(up_conv4)
        up_conv4_l = self.lrelu_only(up_conv4)

        prog_att_1_1 = self.conv_only_1(up_conv4_l)
        prog_att_1_1 = self.conv_only_1(prog_att_1_1)

        prog_att_1_2 = self.conv_only_1(f4)
        att_1 = self.conv_bottleneck_sigmoid(prog_att_1_2)

        prog_att_1_1_m = prog_att_1_1 * att_1
        up_conv4_a = up_conv4_l + prog_att_1_1_m

        f4_c = torch.cat((f4, up_conv4_a), dim=1)
        f4_att = self.conv_only_2(f4_c)
        f4_att = self.conv_only_1(f4_att)

        prog_att_2_1 = self.conv_only_1(up_conv4_a)
        prog_att_2_1 = self.conv_only_1(prog_att_2_1)

        prog_att_2_2 = self.conv_only_1(f4_att)
        att_2 = self.conv_bottleneck_sigmoid(prog_att_2_2)

        prog_att_2_1_m = prog_att_2_1 * att_2
        up_conv4_a_2 = up_conv4_a + prog_att_2_1_m

        f4_c_2 = torch.cat((f4_att, up_conv4_a_2), dim=1)
        f4_att_2 = self.conv_only_2(f4_c_2)
        f4_att_2 = self.conv_only_1(f4_att_2)

        prog_att_3_1 = self.conv_only_1(up_conv4_a_2)
        prog_att_3_1 = self.conv_only_1(prog_att_3_1)

        prog_att_3_2 = self.conv_only_1(f4_att_2)
        att_3 = self.conv_bottleneck_sigmoid(prog_att_3_2)

        prog_att_3_1_m = prog_att_3_1 * att_3
        up_conv4_a_3 = up_conv4_a_2 * prog_att_3_1_m
        up_conv4_a_3 = up_conv4_a_3 + up_conv4_without

        f4_a = up_conv4_a_3 + f4_att_2
        f5 = self.upsample_128_64_lrelu(f4_a)
        f6 = self.upsample_only_narrow(f5)

        up_conv3_c = torch.cat((up_conv4_a_3, f4_a), dim=1)
        up_conv3 = self.upsample_256_128_lrelu(up_conv3_c)

        up_conv2_c = torch.cat((up_conv3, f5), dim=1)
        up_conv2 = self.upsample_192_64_lrelu(up_conv2_c)

        up_conv1_c = torch.cat((up_conv2, f6), dim=1)
        up_conv1 = self.conv2d_transpose(up_conv1_c)
        up_conv1 = self.lrelu_only(up_conv1)

        hr = self.main_branch_tail(up_conv1)

        return hr


class FMFBlock(nn.Module):
    def __init__(self, n_feats=64, n_attributes=40):
        """
        For now this model assumes an input of size 16x16.
        """
        super(FMFBlock, self).__init__()
        self.n_attributes = n_attributes

        self.image_encoder_1 = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16*n_feats, 4*n_attributes),
            nn.PReLU(),
            nn.Linear(4*n_attributes, n_attributes)
        )

        self.image_encoder_2 = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, 1, 1),
            nn.PReLU(),
            Conv2dSame(n_feats, 2*n_feats, 2, 2),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            Conv2dSame(2*n_feats, 4*n_feats, 2, 2),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            Conv2dSame(4*n_feats, 8*n_feats, 2, 2),
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(32*n_feats, 4*n_attributes),
            nn.PReLU(),
            nn.Linear(4*n_attributes, n_attributes)
        )

        self.image_encoder_3 = nn.Sequential(
            nn.Conv2d(3, n_feats, 5, 1, 2),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 5, 1, 2),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_feats, 2*n_feats, 5, 1, 2),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 5, 1, 2),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32*n_feats, 4*n_attributes),
            nn.PReLU(),
            nn.Linear(4*n_attributes, n_attributes)
        )

        self.image_encoder_4 = nn.Sequential(
            nn.Conv2d(3, n_feats, 5, 1, 2),
            nn.PReLU(),
            Conv2dSame(n_feats, 2*n_feats, 2, 2),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 5, 1, 2),
            nn.PReLU(),
            Conv2dSame(2*n_feats, 4*n_feats, 2, 2),
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(64*n_feats, 4*n_attributes),
            nn.PReLU(),
            nn.Linear(4*n_attributes, n_attributes)
        )

        self.image_encoder_5 = nn.Sequential(
            nn.Conv2d(3, n_feats, 7, 1, 3),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 7, 1, 3),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_feats, 2*n_feats, 7, 1, 3),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 7, 1, 3),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32*n_feats, 4*n_attributes),
            nn.PReLU(),
            nn.Linear(4*n_attributes, n_attributes)
        )

        self.image_encoder_6 = nn.Sequential(
            nn.Conv2d(3, n_feats, 7, 1, 3),
            nn.PReLU(),
            Conv2dSame(n_feats, 2*n_feats, 2, 2),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 7, 1, 3),
            nn.PReLU(),
            Conv2dSame(2*n_feats, 4*n_feats, 2, 2),
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(64*n_feats, 4*n_attributes),
            nn.PReLU(),
            nn.Linear(4*n_attributes, n_attributes)
        )

        self.image_encoder_7 = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, 1, 2, 2),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 2, 2),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_feats, 2*n_feats, 3, 1, 2, 2),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 2, 2),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32*n_feats, 4*n_attributes),
            nn.PReLU(),
            nn.Linear(4*n_attributes, n_attributes)
        )

        self.image_encoder_8 = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, 1, 2, 2),
            nn.PReLU(),
            Conv2dSame(n_feats, 2*n_feats, 2, 2),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 2, 2),
            nn.PReLU(),
            Conv2dSame(2*n_feats, 4*n_feats, 2, 2),
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(64*n_feats, 4*n_attributes),
            nn.PReLU(),
            nn.Linear(4*n_attributes, n_attributes)
        )

        self.image_encoder_9 = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, 1, 3, 3),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 3, 3),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_feats, 2*n_feats, 3, 1, 3, 3),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 3, 3),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32*n_feats, 4*n_attributes),
            nn.PReLU(),
            nn.Linear(4*n_attributes, n_attributes)
        )

        self.image_encoder_10 = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, 1, 3, 3),
            nn.PReLU(),
            Conv2dSame(n_feats, 2*n_feats, 2, 2),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 3, 3),
            nn.PReLU(),
            Conv2dSame(2*n_feats, 4*n_feats, 2, 2),
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(64*n_feats, 4*n_attributes),
            nn.PReLU(),
            nn.Linear(4*n_attributes, n_attributes)
        )

        self.attributes_encoder_1 = nn.Sequential(
            nn.Linear(n_attributes, 4*n_attributes),
            nn.PReLU(),
            nn.Linear(4*n_attributes, 8*n_attributes),
            nn.PReLU(),
            nn.Linear(8*n_attributes, 4*n_attributes),
            nn.PReLU(),
            nn.Linear(4*n_attributes, n_attributes),
        )

        self.attributes_encoder_2 = nn.Sequential(
            nn.Linear(n_attributes, n_attributes // 2),
            nn.PReLU(),
            nn.Linear(n_attributes // 2, n_attributes // 4),
            nn.PReLU(),
            nn.Linear(n_attributes // 4, n_attributes // 2),
            nn.PReLU(),
            nn.Linear(n_attributes // 2, n_attributes),
        )

        self.channel_expand = nn.Sequential(
            nn.Conv2d(60, 4*n_attributes, 1, 1)
        )

        self.gap_fused_vector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.squeezed_fused_vector = nn.Sequential(
            nn.Conv2d(4*n_attributes, n_attributes, 1, 1),
            nn.Conv2d(n_attributes, n_attributes // 2, 1, 1),
            nn.Conv2d(n_attributes // 2, 1, 1, 1),
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(n_attributes*n_attributes, 4*n_attributes)
        )

    def forward(self, x, metadata):
        x_1 = self.image_encoder_1(x)
        x_2 = self.image_encoder_2(x)
        x_3 = self.image_encoder_3(x)
        x_4 = self.image_encoder_4(x)
        x_5 = self.image_encoder_5(x)
        x_6 = self.image_encoder_6(x)
        x_7 = self.image_encoder_7(x)
        x_8 = self.image_encoder_8(x)
        x_9 = self.image_encoder_9(x)
        x_10 = self.image_encoder_10(x)

        m_0 = torch.squeeze(torch.squeeze(metadata, 2), 2)
        m_1 = self.attributes_encoder_1(m_0)
        m_2 = self.attributes_encoder_2(m_0)

        # To combine the metadata with the extracted features
        # the outer product will be carried out.
        # This is normally given by: (n, 1) * (1, n) -> (n, n)
        # But since we work with batches, it will be: (b, n, 1) * (b, 1, n) -> (b, n, n)
        # After the bmm (batch matmul), we unsqueeze to get: (b, 1, n, n)

        x_1_m_0 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_1, 2), torch.unsqueeze(m_0, 1)), 1)
        x_1_m_1 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_1, 2), torch.unsqueeze(m_1, 1)), 1)
        x_1_m_2 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_1, 2), torch.unsqueeze(m_2, 1)), 1)

        x_2_m_0 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_2, 2), torch.unsqueeze(m_0, 1)), 1)
        x_2_m_1 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_2, 2), torch.unsqueeze(m_1, 1)), 1)
        x_2_m_2 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_2, 2), torch.unsqueeze(m_2, 1)), 1)

        x_3_m_0 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_3, 2), torch.unsqueeze(m_0, 1)), 1)
        x_3_m_1 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_3, 2), torch.unsqueeze(m_1, 1)), 1)
        x_3_m_2 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_3, 2), torch.unsqueeze(m_2, 1)), 1)

        x_4_m_0 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_4, 2), torch.unsqueeze(m_0, 1)), 1)
        x_4_m_1 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_4, 2), torch.unsqueeze(m_1, 1)), 1)
        x_4_m_2 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_4, 2), torch.unsqueeze(m_2, 1)), 1)

        x_5_m_0 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_5, 2), torch.unsqueeze(m_0, 1)), 1)
        x_5_m_1 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_5, 2), torch.unsqueeze(m_1, 1)), 1)
        x_5_m_2 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_5, 2), torch.unsqueeze(m_2, 1)), 1)

        x_6_m_0 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_6, 2), torch.unsqueeze(m_0, 1)), 1)
        x_6_m_1 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_6, 2), torch.unsqueeze(m_1, 1)), 1)
        x_6_m_2 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_6, 2), torch.unsqueeze(m_2, 1)), 1)

        x_7_m_0 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_7, 2), torch.unsqueeze(m_0, 1)), 1)
        x_7_m_1 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_7, 2), torch.unsqueeze(m_1, 1)), 1)
        x_7_m_2 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_7, 2), torch.unsqueeze(m_2, 1)), 1)

        x_8_m_0 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_8, 2), torch.unsqueeze(m_0, 1)), 1)
        x_8_m_1 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_8, 2), torch.unsqueeze(m_1, 1)), 1)
        x_8_m_2 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_8, 2), torch.unsqueeze(m_2, 1)), 1)

        x_9_m_0 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_9, 2), torch.unsqueeze(m_0, 1)), 1)
        x_9_m_1 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_9, 2), torch.unsqueeze(m_1, 1)), 1)
        x_9_m_2 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_9, 2), torch.unsqueeze(m_2, 1)), 1)

        x_10_m_0 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_10, 2), torch.unsqueeze(m_0, 1)), 1)
        x_10_m_1 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_10, 2), torch.unsqueeze(m_1, 1)), 1)
        x_10_m_2 = torch.unsqueeze(torch.bmm(torch.unsqueeze(x_10, 2), torch.unsqueeze(m_2, 1)), 1)

        x_m_concat = torch.cat((x_1_m_0, x_1_m_1, x_1_m_2, x_2_m_0, x_2_m_1, x_2_m_2,
                                x_3_m_0, x_3_m_1, x_3_m_2, x_4_m_0, x_4_m_1, x_4_m_2,
                                x_5_m_0, x_5_m_1, x_5_m_2, x_6_m_0, x_6_m_1, x_6_m_2,
                                x_7_m_0, x_7_m_1, x_7_m_2, x_8_m_0, x_8_m_1, x_8_m_2,
                                x_9_m_0, x_9_m_1, x_9_m_2, x_10_m_0, x_10_m_1, x_10_m_2), dim=1)

        batch_size = x_m_concat.size(dim=0)
        x_m_channel_num = x_m_concat.size(dim=1)

        diag_mat = torch.unsqueeze(torch.unsqueeze(torch.eye(self.n_attributes, device=x_m_concat.get_device()), 0), 0)
        diag_mat_r = diag_mat.repeat(batch_size, x_m_channel_num, 1, 1)
        diag_mat_r_shifted = diag_mat_r + 0.1  # Maybe make the shift value a user option?

        # This is to give more weighting to the diagonal and less weight to everything else
        x_m_concat_weighted_diag = (x_m_concat + diag_mat_r) * diag_mat_r_shifted

        x_m_concat_all = torch.cat((x_m_concat, x_m_concat_weighted_diag), dim=1)

        x_m_expanded = self.channel_expand(x_m_concat_all)

        x_m_gap_vector = self.gap_fused_vector(x_m_expanded)
        x_m_squeezed_vector = self.squeezed_fused_vector(x_m_expanded)

        x_m_vector_concat = torch.cat((x_m_gap_vector, x_m_squeezed_vector), dim=1)
        x_m_vector_concat_r = torch.unsqueeze(torch.unsqueeze(x_m_vector_concat, 2), 3)

        return x_m_vector_concat_r


class ResidualDenseBlock_4C(nn.Module):
    def __init__(self, in_feats=64, n_feats=64, skip_weight=0.2):
        """
        This is an RRDB-style block
        """
        super(ResidualDenseBlock_4C, self).__init__()
        self.skip_weight = skip_weight

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_feats, n_feats, 3, 1, 1),
            nn.PReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_feats + n_feats, n_feats, 3, 1, 1),
            nn.PReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_feats + n_feats * 2, n_feats, 3, 1, 1),
            nn.PReLU()
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_feats + n_feats * 3, n_feats, 3, 1, 1),
            nn.PReLU()
        )

    def forward(self, x):
        x_1 = self.conv_1(x)
        x_1_c = torch.cat((x, x_1), dim=1)

        x_2 = self.conv_2(x_1_c)
        x_2_c = torch.cat((x, x_1, x_2), dim=1)

        x_3 = self.conv_3(x_2_c)
        x_3_c = torch.cat((x, x_1, x_2, x_3), dim=1)

        x_4 = self.conv_4(x_3_c)

        out = x_4 * self.skip_weight + x

        return out


class FMFResidualDenseNet(nn.Module):
    def __init__(self, n_attributes=40, in_feats=64, n_feats=64, skip_weight=0.2, latent_dim_size_factor=1.0, use_meta_attention=True):
        """
        This is a concept for the super-resolution model that uses the FMF Block.
        The idea is to pass the fused face-meta information both into a latent vector and as an attention vector.
        """
        super(FMFResidualDenseNet, self).__init__()
        self.n_feats = n_feats
        self.use_meta_attention = use_meta_attention
        self.latent_dim_size_factor = latent_dim_size_factor

        self.fmf_block = FMFBlock(n_attributes=n_attributes)

        self.fmf_meta_attention = nn.Sequential(
            nn.Conv2d(8*n_attributes, 6*n_attributes, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(6*n_attributes, 4*n_attributes, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(4*n_attributes, 3*n_attributes, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(3*n_attributes, 2*n_attributes, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(2*n_attributes, n_feats, 1, 1, 0),
            nn.Sigmoid(),
        )

        self.main_branch_head = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, 1, 1),
            nn.PReLU()
        )

        self.main_branch_tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(n_feats, 3, 1, 1)
        )

        self.main_branch_residual_dense_body = nn.Sequential(
            ResidualDenseBlock_4C(in_feats=in_feats, n_feats=n_feats, skip_weight=skip_weight)
        )

        self.main_branch_upsample_block = nn.Sequential(
            nn.Conv2d(n_feats, 2*2*n_feats, 1, 1),
            nn.PReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(n_feats, 2*n_feats, 1, 1),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, n_feats, 1, 1),
            nn.PReLU(),
        )

        # 16 x 16 Encoder-Decoder
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, int(4*latent_dim_size_factor*n_feats), 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(int(4*latent_dim_size_factor*n_feats), 4*n_feats, 2, 2),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.ConvTranspose2d(4*n_feats, 2*n_feats, 2, 2),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.ConvTranspose2d(2*n_feats, n_feats, 2, 2),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
        )

        # 32 x 32 Encoder-Decoder
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, int(4*latent_dim_size_factor*n_feats), 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(int(4*latent_dim_size_factor*n_feats), 4*n_feats, 2, 2),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.ConvTranspose2d(4*n_feats, 4*n_feats, 2, 2),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.ConvTranspose2d(4*n_feats, 2*n_feats, 2, 2),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.ConvTranspose2d(2*n_feats, n_feats, 2, 2),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
        )

        # 64 x 64 Encoder-Decoder
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, int(4*latent_dim_size_factor*n_feats), 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(int(4*latent_dim_size_factor*n_feats), 4*n_feats, 2, 2),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.ConvTranspose2d(4*n_feats, 4*n_feats, 2, 2),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.ConvTranspose2d(4*n_feats, 4*n_feats, 2, 2),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.ConvTranspose2d(4*n_feats, 2*n_feats, 2, 2),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.ConvTranspose2d(2*n_feats, n_feats, 2, 2),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
        )

        self.encoder_decoder_latent_dim = nn.Sequential(
            # (2 x 2) = image size
            # 4 * n_feat = number of channels from encoder
            # latent_dim_size_factor = expansion faction
            nn.Linear(int(2*2*4*latent_dim_size_factor*n_feats) + 8*n_attributes, int(2*2*4*latent_dim_size_factor*n_feats))
        )

        self.encoder_decoder_output_concat_adapter = nn.Sequential(
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.Conv2d(2*n_feats, n_feats, 3, 1, 1),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

    def dense_residual_groups(self, x, fmf):
        # Residual Dense Group 1
        fmf_meta_attention_g_1_b_1 = self.fmf_meta_attention(fmf)
        fmf_meta_attention_g_1_b_2 = self.fmf_meta_attention(fmf)

        x_g_1_b_1 = self.main_branch_residual_dense_body(x)
        x_g_1_b_1 = x_g_1_b_1 * fmf_meta_attention_g_1_b_1
        x_g_1_b_2 = self.main_branch_residual_dense_body(x_g_1_b_1)
        x_g_1_b_2 = x_g_1_b_2 * fmf_meta_attention_g_1_b_2
        x_g_1_a = x + (x_g_1_b_2 * 0.2)

        # Residual Dense Group 2
        fmf_meta_attention_g_2_b_1 = self.fmf_meta_attention(fmf)
        fmf_meta_attention_g_2_b_2 = self.fmf_meta_attention(fmf)

        x_g_2_b_1 = self.main_branch_residual_dense_body(x_g_1_a)
        x_g_2_b_1 = x_g_2_b_1 * fmf_meta_attention_g_2_b_1
        x_g_2_b_2 = self.main_branch_residual_dense_body(x_g_2_b_1)
        x_g_2_b_2 = x_g_2_b_2 * fmf_meta_attention_g_2_b_2
        x_g_2_a = x_g_1_a + (x_g_2_b_2 * 0.2)

        # Residual Dense Group 3
        x_g_3_b_1 = self.main_branch_residual_dense_body(x_g_2_a)
        x_g_3_b_2 = self.main_branch_residual_dense_body(x_g_3_b_1)
        x_g_3_a = x_g_2_a + (x_g_3_b_2 * 0.2)

        return x_g_1_a, x_g_2_a, x_g_3_a

    def dense_residual_groups_no_meta_attention(self, x):
        # Residual Dense Group 1
        x_g_1_b_1 = self.main_branch_residual_dense_body(x)
        x_g_1_b_2 = self.main_branch_residual_dense_body(x_g_1_b_1)
        x_g_1_a = x + (x_g_1_b_2 * 0.2)

        # Residual Dense Group 2
        x_g_2_b_1 = self.main_branch_residual_dense_body(x_g_1_a)
        x_g_2_b_2 = self.main_branch_residual_dense_body(x_g_2_b_1)
        x_g_2_a = x_g_1_a + (x_g_2_b_2 * 0.2)

        # Residual Dense Group 3
        x_g_3_b_1 = self.main_branch_residual_dense_body(x_g_2_a)
        x_g_3_b_2 = self.main_branch_residual_dense_body(x_g_3_b_1)
        x_g_3_a = x_g_2_a + (x_g_3_b_2 * 0.2)

        return x_g_1_a, x_g_2_a, x_g_3_a

    def forward(self, x, metadata):
        fmf = self.fmf_block(x, metadata)
        x_1 = self.main_branch_head(x)

        # 16 x 16
        if self.use_meta_attention:
            _, _, x_1_g_3 = self.dense_residual_groups(x_1, fmf)
        else:
            _, _, x_1_g_3 = self.dense_residual_groups_no_meta_attention(x_1)

        # Encoder-Decoder Block - x_1
        x_1_enc = self.encoder_1(x_1_g_3)
        x_1_enc_c = torch.cat((x_1_enc, torch.squeeze(torch.squeeze(fmf, 2), 2)), dim=1)  # Concatenate FMF features with latent dimension
        x_1_latent_dim = self.encoder_decoder_latent_dim(x_1_enc_c)
        x_1_latent_dim_r = torch.reshape(x_1_latent_dim, (-1, int(4*self.latent_dim_size_factor*self.n_feats), 2, 2))
        x_1_dec = self.decoder_1(x_1_latent_dim_r)

        # Concatenate Residual Block Features and Encoder-Decoder Features - x_1
        x_1_enc_dec_res_c = torch.cat((x_1_g_3, x_1_dec), dim=1)
        x_1_adapter = self.encoder_decoder_output_concat_adapter(x_1_enc_dec_res_c)
        x_1_pre_up_res = self.main_branch_residual_dense_body(x_1_adapter)

        ##############################################################
        # 32 x 32
        x_2 = self.main_branch_upsample_block(x_1_pre_up_res)
        _, _, x_2_g_3 = self.dense_residual_groups(x_2, fmf)

        # Encoder-Decoder Block - x_2
        x_2_enc = self.encoder_2(x_2_g_3)
        x_2_enc_c = torch.cat((x_2_enc, torch.squeeze(torch.squeeze(fmf, 2), 2)), dim=1)  # Concatenate FMF features with latent dimension
        x_2_latent_dim = self.encoder_decoder_latent_dim(x_2_enc_c)
        x_2_latent_dim_r = torch.reshape(x_2_latent_dim, (-1, int(4*self.latent_dim_size_factor*self.n_feats), 2, 2))
        x_2_dec = self.decoder_2(x_2_latent_dim_r)

        # Concatenate Residual Block Features and Encoder-Decoder Features - x_2
        x_2_enc_dec_res_c = torch.cat((x_2_g_3, x_2_dec), dim=1)
        x_2_adapter = self.encoder_decoder_output_concat_adapter(x_2_enc_dec_res_c)
        x_2_pre_up_res = self.main_branch_residual_dense_body(x_2_adapter)

        ##############################################################
        # 64 x 64
        x_3 = self.main_branch_upsample_block(x_2_pre_up_res)
        _, _, x_3_g_3 = self.dense_residual_groups(x_3, fmf)

        # Encoder-Decoder Block - x_3
        x_3_enc = self.encoder_3(x_3_g_3)
        x_3_enc_c = torch.cat((x_3_enc, torch.squeeze(torch.squeeze(fmf, 2), 2)), dim=1)  # Concatenate FMF features with latent dimension
        x_3_latent_dim = self.encoder_decoder_latent_dim(x_3_enc_c)
        x_3_latent_dim_r = torch.reshape(x_3_latent_dim, (-1, int(4*self.latent_dim_size_factor*self.n_feats), 2, 2))
        x_3_dec = self.decoder_3(x_3_latent_dim_r)

        # Concatenate Residual Block Features and Encoder-Decoder Features - x_3
        x_3_enc_dec_res_c = torch.cat((x_3_g_3, x_3_dec), dim=1)
        x_3_adapter = self.encoder_decoder_output_concat_adapter(x_3_enc_dec_res_c)
        x_3_pre_up_res = self.main_branch_residual_dense_body(x_3_adapter)

        ##############################################################
        # 128 x 128
        x_4 = self.main_branch_upsample_block(x_3_pre_up_res)
        x_4_g_1_b_1 = self.main_branch_residual_dense_body(x_4)
        x_4_g_1_b_2 = self.main_branch_residual_dense_body(x_4_g_1_b_1)

        out = self.main_branch_tail(x_4_g_1_b_2)

        return out

    def forensic(self, x, metadata):

        fmf = self.fmf_block(x, metadata)
        out = self.forward(x, metadata)

        return out, fmf
