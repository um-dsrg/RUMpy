import math

import torch
import torch.nn.functional as F
from torch import nn

from rumpy.SISR.models.face_attributes_gan_models.common_blocks import *


class FaceSRAttributesDiscriminatorNet(nn.Module):
    """
    Based on implementation in https://github.com/XinYuANU/FaceAttr/blob/master/train_encoder_decoder_Stack_skip_perception.lua
    """

    def __init__(self, n_feats=32, n_attributes=18, use_attribute_encoder=False):
        super(FaceSRAttributesDiscriminatorNet, self).__init__()

        self.discriminator_first_layer = nn.Conv2d(3, n_feats, 5, 1, 2)

        # First Discriminator Network
        self.discriminator_head = nn.Sequential(
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats*2, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )

        # Explanation of what the Replicate command does: https://github.com/torch/nn/blob/master/doc/simple.md#nn.Replicate
        # Topic Discussions:
        # https://discuss.pytorch.org/t/torch-equivalent-code-for-pytorch-with-nn-replicate/10398/2
        # https://discuss.pytorch.org/t/concatenate-layer-output-with-additional-input-data/20462/48

        # As a basic idea, when nn.Replicate(32,3,4) is used, this means that the input
        # will be repeated 32 times along dimension 3 (in BxCxHxW, 3rd dim = H).
        # The number 4 signifies ndim, which is the number of non-batch dimensions.
        # This command is used to turn the attributes of size 18x1 into 18x32x32.

        self.discriminator_body = nn.Sequential(
            nn.Conv2d((n_feats*2)+n_attributes, n_feats*4, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(n_feats*4, n_feats*8, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Flatten(),
            nn.Linear(8*8*8*n_feats, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        # Adding this to see if perhaps modifying the attributes before concatenation will help
        self.metadata_layer = nn.Sequential(nn.Identity())

        if use_attribute_encoder:
            self.metadata_layer = nn.Sequential(
                nn.Conv2d(n_attributes, n_attributes*2, 1, 1),
                nn.Conv2d(n_attributes*2, n_attributes, 1, 1),
            )

    def forward(self, x, metadata):
        f = self.discriminator_first_layer(x)
        x = self.discriminator_head(f)

        # Expand the metadata to BxCxHxW
        # NOTE: This is only needed if the expansion isn't done in the handler (which it is)
        # metadata = torch.unsqueeze(metadata, 2)
        # metadata = torch.unsqueeze(metadata, 3)

        # Repeat the metadata to fill out the 32x32 size of an image
        # This should end up with B x n_attributes x 32 x 32
        metadata = metadata.repeat(1, 1, 32, 32)
        metadata_out = self.metadata_layer(metadata)

        combined = torch.cat((x, metadata_out), dim=1)

        out = self.discriminator_body(combined)

        return out


class AGAGANDiscriminatorNet(nn.Module):
    """
    Based on implementation in https://github.com/NoviceMAn-prog/AGA-GAN/blob/main/agagan_unet.py
    """

    def __init__(self, n_feats=32, n_attributes=38):
        super(AGAGANDiscriminatorNet, self).__init__()

        self.attributes_dense_block = nn.Sequential(
            nn.Linear(n_attributes, 768),
            nn.LeakyReLU(0.25)
        )

        self.attributes_conv_block = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.25),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.25),
            nn.ConvTranspose2d(n_feats, n_feats*2, 4, 2, 1),  # The padding here is hardcoded since we know the input size
            nn.LeakyReLU(0.25)
        )

        self.main_branch_head = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.25),
            # nn.Conv2d(n_feats, n_feats, 4, 2, 'same'),
            Conv2dSame(n_feats, n_feats, 4, 2),  # Conv2dSame calculates the 'same' padding for stride of 2
            nn.LeakyReLU(0.25),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.25),
            # nn.Conv2d(n_feats, n_feats*2, 4, 2, 'same'),
            Conv2dSame(n_feats, n_feats*2, 4, 2),  # Conv2dSame calculates the 'same' padding for stride of 2
            nn.LeakyReLU(0.25)
        )

        self.main_branch_body = nn.Sequential(
            nn.Conv2d(n_feats*4, n_feats*2, 3, 1, 1),
            nn.LeakyReLU(0.25),
            # nn.Conv2d(n_feats*2, n_feats*4, 4, 2, 'same'),
            Conv2dSame(n_feats*2, n_feats*4, 4, 2),  # Conv2dSame calculates the 'same' padding for stride of 2
            nn.LeakyReLU(0.25),
            nn.Conv2d(n_feats*4, n_feats*4, 3, 1, 1),
            nn.LeakyReLU(0.25),
            # nn.Conv2d(n_feats*4, n_feats*3, 4, 2, 'same'),
            Conv2dSame(n_feats*4, n_feats*3, 4, 2),  # Conv2dSame calculates the 'same' padding for stride of 2
            nn.LeakyReLU(0.25),
            nn.Conv2d(n_feats*3, n_feats*3, 3, 1, 1),
            nn.LeakyReLU(0.25),
        )

        self.main_branch_tail = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6144, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, metadata):
        # Convert from B x C x 1 x 1 to B x C for Dense/Linear layer
        metadata_r = torch.squeeze(metadata)
        att = self.attributes_dense_block(metadata_r)
        att_r = torch.reshape(att, (-1, 3, 16, 16))
        att_f = self.attributes_conv_block(att_r)

        conv_1_4 = self.main_branch_head(x)
        conv_1_4_c = torch.cat((conv_1_4, att_f), dim=1)

        conv_5_9 = self.main_branch_body(conv_1_4_c)

        out = self.main_branch_tail(conv_5_9)

        return out


class FMFDiscriminator(nn.Module):
    """
    Discriminator for the FMF Network.
    Unlike the others, for now this will not include attributes as those will have a separate model.
    """

    def __init__(self, n_feats=64, use_sigmoid=True):
        super(FMFDiscriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16*n_feats, 8*n_feats),
            nn.PReLU(),
            nn.Linear(8*n_feats, 1),
        )

        self.final_layer = nn.Sequential(nn.Identity())

        if use_sigmoid:
            self.final_layer = nn.Sequential(
                nn.Sigmoid()
            )

    def forward(self, x):
        x_1 = self.discriminator(x)
        out = self.final_layer(x_1)

        return out


class FMFAttributeDiscriminator(nn.Module):
    """
    Discriminator for the attributes of the FMF Network.
    """

    def __init__(self, n_feats=64, n_attributes=40, use_sigmoid=True):
        super(FMFAttributeDiscriminator, self).__init__()

        self.discriminator = nn.Sequential(
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
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(2*n_feats, 2*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(4*n_feats, 4*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4*n_feats, 8*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(8*n_feats, 8*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(8*n_feats, 8*n_feats, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32*n_feats, 8*n_feats),
            nn.PReLU(),
            nn.Linear(8*n_feats, n_attributes),
        )

        self.final_layer = nn.Sequential(nn.Identity())

        if use_sigmoid:
            self.final_layer = nn.Sequential(
                nn.Sigmoid()
            )

    def forward(self, x):
        out = self.discriminator(x)

        return out
