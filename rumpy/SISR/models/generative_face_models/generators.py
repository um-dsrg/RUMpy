import torch
import torch.nn.functional as F
from rumpy.SISR.models.face_attributes_gan_models.common_blocks import *
from torch import nn


class GANGenerator(nn.Module):
    """
    Based on the tutorial in the 'Machine Learning Mastery' book.
    """

    def __init__(self, latent_dim=100):
        super(GANGenerator, self).__init__()
        self.generator_head = nn.Sequential(
            nn.Linear(latent_dim, 128*5*5),
            nn.LeakyReLU(0.2)
        )

        self.generator_body = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 5x5 -> 10x10
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 10x10 -> 20x20
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 20x20 -> 40x40
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 40x40 -> 80x80
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 3, 5, 1, 2),
            nn.Tanh()
        )

    def forward(self, x):
        x_1 = self.generator_head(x)
        x_1_r = torch.reshape(x_1, (-1, 128, 5, 5))

        out = self.generator_body(x_1_r)

        return out
