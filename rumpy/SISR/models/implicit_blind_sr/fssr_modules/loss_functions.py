# all extracted from https://github.com/ManuelFritsche/real-world-sr, lpips modified to match recent implementation
import torch
from torch import nn
import random
from lpips import lpips


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, stride=1, padding=4):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, padding=True, include_pad=True, gaussian=False):
        super(FilterLow, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img


def generator_loss(labels, wasserstein=False, weights=None):
    if not isinstance(labels, list):
        labels = (labels,)
    if weights is None:
        weights = [1.0 / len(labels)] * len(labels)
    loss = 0.0
    for label, weight in zip(labels, weights):
        if wasserstein:
            loss += weight * torch.mean(-label)
        else:
            loss += weight * torch.mean(-torch.log(label + 1e-8))
    return loss


def discriminator_loss(reals, fakes, wasserstein=False, grad_penalties=None, weights=None):
    if not isinstance(reals, list):
        reals = (reals,)
    if not isinstance(fakes, list):
        fakes = (fakes,)
    if weights is None:
        weights = [1.0 / len(fakes)] * len(fakes)
    loss = 0.0
    if wasserstein:
        if not isinstance(grad_penalties, list):
            grad_penalties = (grad_penalties,)
        for real, fake, weight, grad_penalty in zip(reals, fakes, weights, grad_penalties):
            loss += weight * (-real.mean() + fake.mean() + grad_penalty)
    else:
        for real, fake, weight in zip(reals, fakes, weights):
            loss += weight * (-torch.log(real + 1e-8).mean() - torch.log(1 - fake + 1e-8).mean())
    return loss


class PerceptualLossLPIPS(nn.Module):
    def __init__(self, device):
        super(PerceptualLossLPIPS, self).__init__()
        self.loss_network = lpips.LPIPS(net='vgg').to(device)

    def forward(self, x, y):
        return self.loss_network.forward(x, y, normalize=True).mean()



class PerceptualLoss(nn.Module):
    def __init__(self, device, rotations=False, flips=False):
        super(PerceptualLoss, self).__init__()
        self.loss = PerceptualLossLPIPS(device)
        self.rotations = rotations
        self.flips = flips

    def forward(self, x, y):
        if self.rotations:
            k_rot = random.choice([-1, 0, 1])
            x = torch.rot90(x, k_rot, [2, 3])
            y = torch.rot90(y, k_rot, [2, 3])
        if self.flips:
            if random.choice([True, False]):
                x = torch.flip(x, (2,))
                y = torch.flip(y, (2,))
            if random.choice([True, False]):
                x = torch.flip(x, (3,))
                y = torch.flip(y, (3,))
        return self.loss(x, y)

class GeneratorLoss(nn.Module):
    def __init__(self, recursions=1, stride=1, kernel_size=5, use_perceptual_loss=True, wgan=False, w_col=1,
                 w_tex=0.005, w_per=0.01, gaussian=False, lpips_rot_flip=False, device=torch.device('cpu'), **kwargs):
        super(GeneratorLoss, self).__init__()
        self.pixel_loss = nn.L1Loss().to(device=device)
        self.color_filter = FilterLow(recursions=recursions, stride=stride, kernel_size=kernel_size, padding=False,
                                      gaussian=gaussian).to(device=device)
        self.perceptual_loss = PerceptualLoss(device, rotations=lpips_rot_flip, flips=lpips_rot_flip)
        self.use_perceptual_loss = use_perceptual_loss
        self.wasserstein = wgan
        self.w_col = w_col
        self.w_tex = w_tex
        self.w_per = w_per
        self.last_tex_loss = 0
        self.last_per_loss = 0
        self.last_col_loss = 0
        self.last_mean_loss = 0

    def forward(self, tex_labels, out_images, target_images):
        # Adversarial Texture Loss
        self.last_tex_loss = generator_loss(tex_labels, wasserstein=self.wasserstein)
        # Perception Loss
        self.last_per_loss = self.perceptual_loss(out_images, target_images)
        # Color Loss
        self.last_col_loss = self.color_loss(out_images, target_images)
        loss = self.w_col * self.last_col_loss + self.w_tex * self.last_tex_loss
        if self.use_perceptual_loss:
            loss += self.w_per * self.last_per_loss
        return loss

    def color_loss(self, x, y):
        return self.pixel_loss(self.color_filter(x), self.color_filter(y))

    def rgb_loss(self, x, y):
        return self.pixel_loss(x.mean(3).mean(2), y.mean(3).mean(2))

    def mean_loss(self, x, y):
        return self.pixel_loss(x.view(x.size(0), -1).mean(1), y.view(y.size(0), -1).mean(1))

