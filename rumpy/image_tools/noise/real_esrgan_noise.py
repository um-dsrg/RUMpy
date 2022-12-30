# all code here modified from https://github.com/xinntao/BasicSR/blob/master/basicsr/data/degradations.py
import cv2
import numpy as np
import torch

from torchvision.transforms.functional_tensor import rgb_to_grayscale


def format_scalar_or_vector_pt(pt_variable):
    """
    Formats a pytorch variable into either a scalar or a numpy vector output.
    :param pt_variable: Tensor array (single value or vector)
    :return: Numpy array or scalar (float)
    """

    return pt_variable.numpy() if len(pt_variable) > 1 else float(pt_variable)

# ----------------------- Gaussian Noise ----------------------- #
def generate_gaussian_noise(img, sigma=10, gray_noise=False):
    """Generate Gaussian noise.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        sigma (float): Noise scale (measured in range 255). Default: 10.
    Returns:
        (Numpy array): Returned noisy image, shape (h, w, c), range[0, 1],
            float32.
    """
    if gray_noise:
        noise = np.float32(np.random.randn(*(img.shape[0:2]))) * sigma / 255.
        noise = np.expand_dims(noise, axis=2).repeat(3, axis=2)
    else:
        noise = np.float32(np.random.randn(*(img.shape))) * sigma / 255.
    return noise


def add_gaussian_noise(img, sigma=10, clip=True, rounds=False, gray_noise=False):
    """Add Gaussian noise.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        sigma (float): Noise scale (measured in range 255). Default: 10.
    Returns:
        (Numpy array): Returned noisy image, shape (h, w, c), range[0, 1],
            float32.
    """
    noise = generate_gaussian_noise(img, sigma, gray_noise)
    out = img + noise
    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


def generate_gaussian_noise_pt(img, sigma=10, gray_noise=0):
    """Add Gaussian noise (PyTorch version).
    Args:
        img (Tensor): Shape (b, c, h, w), range[0, 1], float32.
        scale (float | Tensor): Noise scale. Default: 1.0.
    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    b, _, h, w = img.size()

    if not isinstance(sigma, (float, int)):
        sigma = sigma.view(b, 1, 1, 1)

    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0

    if cal_gray_noise:
        noise_gray = torch.randn(*img.size()[2:4], dtype=img.dtype, device=img.device) * sigma / 255.
        noise_gray = noise_gray.view(b, 1, h, w)
        return noise_gray.expand(b, 3, h, w)
    else:
        # Colour noise
        return torch.randn(*img.size(), dtype=img.dtype, device=img.device) * sigma / 255.


def add_gaussian_noise_pt(img, sigma=10, gray_noise=0, clip=True, rounds=False):
    """Add Gaussian noise (PyTorch version).
    Args:
        img (Tensor): Shape (b, c, h, w), range[0, 1], float32.
        scale (float | Tensor): Noise scale. Default: 1.0.
    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    # noise = generate_gaussian_noise_pt(img, sigma, gray_noise)
    noise = generate_gaussian_noise_pt(img, sigma, None)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


# ----------------------- Random Gaussian Noise ----------------------- #
def random_generate_gaussian_noise(img, sigma_range=(0, 10), gray_prob=0):
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    if np.random.uniform() < gray_prob:
        gray_noise = True
    else:
        gray_noise = False
    return generate_gaussian_noise(img, sigma, gray_noise)


def random_add_gaussian_noise(img, sigma_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_gaussian_noise(img, sigma_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


def random_generate_gaussian_noise_pt(img, sigma_range=(0, 10), gray_prob=0):
    sigma = torch.rand(
        img.size(0), dtype=img.dtype, device=img.device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()

    metadata = {'gaussian_noise_scale': format_scalar_or_vector_pt(sigma), 'gray_noise': format_scalar_or_vector_pt(gray_noise),
                'poisson_noise_scale': 0.0}

    return generate_gaussian_noise_pt(img, sigma, gray_noise), metadata


def random_add_gaussian_noise_pt(img, sigma_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise, metadata = random_generate_gaussian_noise_pt(img, sigma_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out, metadata, noise


# ----------------------- Poisson (Shot) Noise ----------------------- #


def generate_poisson_noise(img, scale=1.0, gray_noise=False):
    """Generate poisson noise.
    Ref: https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L37-L219
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        scale (float): Noise scale. Default: 1.0.
        gray_noise (bool): Whether generate gray noise. Default: False.
    Returns:
        (Numpy array): Returned noisy image, shape (h, w, c), range[0, 1],
            float32.
    """
    if gray_noise:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # round and clip image for counting vals correctly
    img = np.clip((img * 255.0).round(), 0, 255) / 255.
    vals = len(np.unique(img))
    vals = 2**np.ceil(np.log2(vals))
    out = np.float32(np.random.poisson(img * vals) / float(vals))
    noise = out - img
    if gray_noise:
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    return noise * scale


def add_poisson_noise(img, scale=1.0, clip=True, rounds=False, gray_noise=False):
    """Add poisson noise.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        scale (float): Noise scale. Default: 1.0.
        gray_noise (bool): Whether generate gray noise. Default: False.
    Returns:
        (Numpy array): Returned noisy image, shape (h, w, c), range[0, 1],
            float32.
    """
    noise = generate_poisson_noise(img, scale, gray_noise)
    out = img + noise
    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


def generate_poisson_noise_pt(img, scale=1.0, gray_noise=0):
    """Generate a batch of poisson noise (PyTorch version)
    Args:
        img (Tensor): Input image, shape (b, c, h, w), range [0, 1], float32.
        scale (float | Tensor): Noise scale. Number or Tensor with shape (b).
            Default: 1.0.
        gray_noise (float | Tensor): 0-1 number or Tensor with shape (b).
            0 for False, 1 for True. Default: 0.
    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    b, _, h, w = img.size()

    if not isinstance(scale, (float, int)):
        scale = scale.view(b, 1, 1, 1)

    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0

    if cal_gray_noise:
        # convert RGB image to grayscale
        img_gray = rgb_to_grayscale(img, num_output_channels=1)
        # round and clip image for counting vals correctly
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.

        # use for-loop to get the unique values for each sample
        vals_list = [len(torch.unique(img_gray[i, :, :, :])) for i in range(b)]
        vals_list = [2**np.ceil(np.log2(vals)) for vals in vals_list]
        vals = img_gray.new_tensor(vals_list).view(b, 1, 1, 1)

        noise_gray = (torch.poisson(img_gray * vals) / vals) - img_gray
        return noise_gray.expand(b, 3, h, w) * scale
    else:
        # round and clip image for counting vals correctly
        img = torch.clamp((img * 255.0).round(), 0, 255) / 255.

        # use for-loop to get the unique values for each sample
        vals_list = [len(torch.unique(img[i, :, :, :])) for i in range(b)]
        vals_list = [2**np.ceil(np.log2(vals)) for vals in vals_list]
        vals = img.new_tensor(vals_list).view(b, 1, 1, 1)

        noise = (torch.poisson(img * vals) / vals) - img
        return noise * scale


def add_poisson_noise_pt(img, scale=1.0, clip=True, rounds=False, gray_noise=0):
    """Add poisson noise to a batch of images (PyTorch version).
    Args:
        img (Tensor): Input image, shape (b, c, h, w), range [0, 1], float32.
        scale (float | Tensor): Noise scale. Number or Tensor with shape (b).
            Default: 1.0.
        gray_noise (float | Tensor): 0-1 number or Tensor with shape (b).
            0 for False, 1 for True. Default: 0.
    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    noise = generate_poisson_noise_pt(img, scale, gray_noise)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


# ----------------------- Random Poisson (Shot) Noise ----------------------- #


def random_generate_poisson_noise(img, scale_range=(0, 1.0), gray_prob=0):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    if np.random.uniform() < gray_prob:
        gray_noise = True
    else:
        gray_noise = False
    return generate_poisson_noise(img, scale, gray_noise)


def random_add_poisson_noise(img, scale_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_poisson_noise(img, scale_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


def random_generate_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0):
    scale = torch.rand(
        img.size(0), dtype=img.dtype, device=img.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()

    metadata = {'gaussian_noise_scale': 0.0,
                'gray_noise': format_scalar_or_vector_pt(gray_noise),
                'poisson_noise_scale': format_scalar_or_vector_pt(scale)}

    return generate_poisson_noise_pt(img, scale, gray_noise), metadata


def random_add_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise, metadata = random_generate_poisson_noise_pt(img, scale_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out, metadata, noise
