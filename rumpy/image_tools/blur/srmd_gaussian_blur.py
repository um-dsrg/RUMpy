# this file has been modified from https://github.com/yuanjunchai/IKC/blob/master/codes/utils/util.py
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import collections
from rumpy.shared_framework.configuration.gpu_check import device_selector


try:
    import accimage
except ImportError:
    accimage = None


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.
    See :class:`~torchvision.transforms.ToPIlImage` for more details.
    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes
    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        if npimg.dtype == np.int16:
            expected_mode = 'I;16'
        if npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(255)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def cal_sigma(sig_x, sig_y, radians):
    D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
    sigma = np.dot(U, np.dot(D, U.T))
    return sigma


def anisotropic_gaussian_kernel(l, sigma_matrix, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def isotropic_gaussian_kernel(l, sigma, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def random_anisotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, scaling=3, l=21, tensor=False):
    pi = np.random.random() * math.pi * 2 - math.pi
    x = np.random.random() * (sig_max - sig_min) + sig_min
    y = np.clip(np.random.random() * scaling * x, sig_min, sig_max)
    sig = cal_sigma(x, y, pi)
    k = anisotropic_gaussian_kernel(l, sig, tensor=tensor)
    return k, (pi, x, y)


def random_isotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, l=21, tensor=False):
    x = np.random.random() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k, x


def stable_isotropic_gaussian_kernel(sig=2.6, l=21, tensor=False):
    x = sig
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def random_gaussian_kernel(l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=False):
    if np.random.random() < rate_iso:
        return random_isotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scaling=scaling, tensor=tensor)


def stable_gaussian_kernel(l=21, sig=2.6, tensor=False):
    return stable_isotropic_gaussian_kernel(sig=sig, l=l, tensor=tensor)


def random_batch_kernel(batch, kernel=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=True, **kwargs):
    batch_kernel = np.zeros((batch, kernel, kernel))
    metadata = []
    for i in range(batch):
        batch_kernel[i], kernel_info = random_gaussian_kernel(l=kernel, sig_min=sig_min, sig_max=sig_max,
                                                              rate_iso=rate_iso,
                                                              scaling=scaling, tensor=False)
        metadata.append(kernel_info)

    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel, metadata


def stable_batch_kernel(batch, l=21, sig=2.6, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    for i in range(batch):
        batch_kernel[i] = stable_gaussian_kernel(l=l, sig=sig, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


def random_batch_noise(batch, high, rate_cln=1.0):
    noise_level = np.random.uniform(size=(batch, 1)) * high
    noise_mask = np.random.uniform(size=(batch, 1))
    noise_mask[noise_mask < rate_cln] = 0
    noise_mask[noise_mask >= rate_cln] = 1
    return noise_level * noise_mask


def b_GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.mul(torch.FloatTensor(np.random.normal(loc=mean, scale=1.0, size=size)), sigma.view(sigma.size() + (1, 1)))
    return torch.clamp(noise + tensor, min=min, max=max)


class BatchSRKernel(object):
    def __init__(self, l=21, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3):
        self.l = l
        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate = rate_iso
        self.scaling = scaling

    def __call__(self, random, batch, tensor=False):
        if random:  # random kernel
            return random_batch_kernel(batch, kernel=self.l, sig_min=self.sig_min, sig_max=self.sig_max,
                                       rate_iso=self.rate, scaling=self.scaling, tensor=tensor)
        else:  # stable kernel
            return stable_batch_kernel(batch, l=self.l, sig=self.sig, tensor=tensor), [self.sig] * batch


def PCA(data, k=2):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.t(X))
    return U[:, :k]  # PCA matrix


class PCAEncoder(object):
    def __init__(self, weight, device=torch.device('cpu')):

        device = device_selector(True, device)

        self.weight = weight #[l^2, k]
        self.size = self.weight.size()
        self.t_size = self.weight.t().size()
        self.weight = self.weight.to(device=device)

    def invert_pca(self, batch_encoded_rep):
        B, k = batch_encoded_rep.size()  # [B, k]
        return torch.bmm(batch_encoded_rep.view((B, 1, k)), self.weight.t().expand((B, ) + self.t_size)).view((B, -1))

    def __call__(self, batch_kernel):  # TODO: why are they not subtracting mean from input data?
        B, D = batch_kernel.size()  # [batch_size, rest of dims]
        return torch.bmm(batch_kernel.view((B, 1, D)), self.weight.expand((B, ) + self.size)).view((B, -1))


class BatchBlur(nn.Module):
    def __init__(self, l=15):
        super(BatchBlur, self).__init__()
        self.l = l
        if l % 2 == 1:
            self.pad = nn.ReflectionPad2d(l // 2)
        else:
            self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        # self.pad = nn.ZeroPad2d(l // 2)

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            input_CBHW = pad.view((1, C * B, H_p, W_p))
            kernel_var = kernel.contiguous().view((B, 1, self.l, self.l)).repeat(1, C, 1, 1).view((B * C, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, groups=B*C).view((B, C, H, W))


class SRMDPreprocessing(object):
    def __init__(self, random=False, para_input=10, kernel=21, noise=False, device=torch.device('cpu'), sig=2.6,
                 sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, rate_cln=0.2, noise_high=0.08):
        """
        Main class for SRMD-style blurring/noise addition.  Also produces a PCA-encoded version of the
        blur kernel used on application.
        Code modified from IKC repo (https://github.com/yuanjunchai/IKC/blob/master/codes/utils/util.py)
        :param random: Set to true to generate randomly-sized blur kernels for each input
        :param para_input: Number of elements to encode from blur kernel using PCA
        :param kernel: Kernel size (int - assumed square)
        :param noise: Set to true to include noise addition after blurring
        :param device: Pytorch GPU or CPU device to use for application of pipeline
        :param sig: Blur kernel sigma to use (if not random)
        :param sig_min: Minimum value of blur kernel sigma to use (if random)
        :param sig_max: Maximum value of blur kernel sigma to use (if random)
        :param rate_iso: Probability of generating an isotropic kernel (set to 1.0 to never generate anisotropic kernels)
        :param scaling:  Maximum ratio of anisotropic dimension scale (actual scale will be randomly selected)
        :param rate_cln: TODO: investigate noise mechanism
        :param noise_high: TODO: investigate noise mechanism
        """

        device = device_selector(True, device)

        if sig is None:  # default value
            sig = 2.6

        self.kernel_gen = BatchSRKernel(l=kernel, sig=sig, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso,
                                        scaling=scaling)  # prepares Gaussian kernel generator
        self.blur = BatchBlur(l=kernel)
        self.para_in = para_input
        self.l = kernel
        self.noise = noise
        self.device = device
        self.rate_cln = rate_cln
        self.noise_high = noise_high
        self.random = random  # random kernel width selection

    def reset_kernel(self, kernel=21, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, random=False):
        self.kernel_gen = BatchSRKernel(l=kernel, sig=sig, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso,
                                        scaling=scaling)
        self.random = random

    def __call__(self, hr_tensor):

        hr_tensor = hr_tensor.unsqueeze(0)
        B, C, H, W = hr_tensor.size()

        b_kernels, metadata = self.kernel_gen(self.random, B, tensor=True)  # kernel generation

        b_kernels = b_kernels.to(self.device)
        # blurring application
        hr_blurred_var = self.blur(hr_tensor.to(self.device), b_kernels)

        # Noise addition
        if self.noise:
            Noise_level = torch.FloatTensor(random_batch_noise(B, self.noise_high, self.rate_cln))
            lr_noised_t = b_GaussianNoising(hr_blurred_var, Noise_level)
        else:
            lr_noised_t = hr_blurred_var

        return lr_noised_t, b_kernels, metadata

