# This code is very similar to VGGStyleDiscriminator128, it has been simplified slightly
# The code here adapted from:
# https://github.com/xinntao/BasicSR/tree/4d34d071218d0e767096eddefa919200d5239936
# https://github.com/lemoner20/SR_2020/blob/master/model.py
# https://ieeexplore-ieee-org.ejournals.um.edu.mt/document/9055090
from torch import nn

class VGGStyleDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, num_in_ch=3, num_feat=64, img_size=256):
        super(VGGStyleDiscriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Conv2d(num_feat, num_feat, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        layers.append(nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Conv2d(num_feat, num_feat, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        layers.append(nn.Conv2d(num_feat, num_feat * 2, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Conv2d(num_feat * 2, num_feat * 2, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        layers.append(nn.Conv2d(num_feat * 2, num_feat * 4, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Conv2d(num_feat * 4, num_feat * 4, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        layers.append(nn.Conv2d(num_feat * 4, num_feat * 8, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Conv2d(num_feat * 8, num_feat * 8, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        layers.append(nn.Flatten())

        new_img_size = img_size // 32

        layers.append(nn.Linear(num_feat * 8 * new_img_size * new_img_size, 1024))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        layers.append(nn.Linear(1024, 256))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        layers.append(nn.Linear(256, 1))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out
