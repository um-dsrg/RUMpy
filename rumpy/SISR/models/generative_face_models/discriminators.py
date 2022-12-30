import torch
from torch import nn

from rumpy.SISR.models.face_attributes_gan_models.common_blocks import *


class GANDiscriminator(nn.Module):
    """
    Based on the tutorial in the 'Machine Learning Mastery' book.
    """

    def __init__(self):
        super(GANDiscriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 128, 5, 1, 2),
            nn.LeakyReLU(0.2),
            Conv2dSame(128, 128, 5, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            Conv2dSame(128, 128, 5, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            Conv2dSame(128, 128, 5, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            Conv2dSame(128, 128, 5, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(5*5*128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.discriminator(x)

        return out
