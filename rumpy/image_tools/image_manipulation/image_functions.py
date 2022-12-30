import random
import math

import PIL.Image
import numpy as np
import torch
from torchvision import transforms
from skimage.filters.rank import entropy
from skimage.morphology import rectangle


###### Scaling ######
def downsample(image, scale, jm=False):
    # TODO: add further downsample options (bilinear, etc)
    """
    Downsamples image according to scale factor, taking into consideration restrictions imposed by JM compression.
    :param image: Input PIL HR image.
    :param scale: Scale factor for downsampling.
    :param jm: Set to True if special consideration for JM compression needs to be made.
    :return: Cropped HR image, downsampled LR image.
    """
    if jm:
        corrected_width = (math.floor(image.width / scale) // 2) * 2  # JM only accepts even dimensions (for unknown reasons)
        corrected_height = (math.floor(image.height / scale) // 2) * 2
    else:
        corrected_width = math.floor(image.width / scale)
        corrected_height = math.floor(image.height / scale)

    r_width = corrected_width * scale  # re-scaling HR image to dimensions which match selected scale
    r_height = corrected_height * scale

    hr_image = CenterCrop(width=r_width, height=r_height)(image)

    lr_image = hr_image.resize((int(math.floor(r_width / scale)), int(math.floor(r_height / scale))), resample=PIL.Image.BICUBIC)  # downsize
    return hr_image, lr_image


def upsample(image, scale):
    # TODO: add further downsample options (bilinear, etc)
    hr_im = image.resize((image.width * scale, image.height * scale), resample=PIL.Image.BICUBIC)
    return hr_im
###### ---- ######

###### Face Cropping ######
def landmark_crop(image, crop_size, landmarks):
    if type(landmarks) == str:
        centroid = (image.width/2, image.height/2)
    else:
        centroid = (landmarks.max(0) + landmarks.min(0))/2
    l_pos, t_pos = centroid[0] - (crop_size[0]/2), centroid[1] - (crop_size[1]/2)
    cropped_image = image.crop((l_pos, t_pos, l_pos + crop_size[0], t_pos + crop_size[1]))

    if type(landmarks) == str:
        scaled_landmarks = landmarks
    else:
        scaled_landmarks = np.copy(landmarks)
        scaled_landmarks[:, 0] = landmarks[:, 0] - l_pos
        scaled_landmarks[:, 1] = landmarks[:, 1] - t_pos

    return cropped_image, scaled_landmarks


def detect_negative_landmarks(landmarks):
    if (landmarks < 0).any():
        return True
    else:
        return False
###### ---- ######

###### YCbCr conversion ######
# TODO: find way to perform conversion using matrix multiplication for faster action
def rgb_to_ycbcr(img, y_only=True, max_val=1, im_type='png'):  # image always expected in C, H, W format
    """
    Converts RGB image to YCbCr.
    :param img: CxHxW Image.
    :param y_only: Set as true to only retrieve luminance values.
    :param max_val: Image dynamic range.
    :param im_type: PNG or JPG format.
    :return:  Converted Image.
    """
    if im_type == 'jpg':

        bias_c = 128.*(max_val/255)

        y = (0.299 * img[0, :, :] + 0.587 * img[1, :, :] + 0.114 * img[2, :, :])

        if y_only:
            return y, None, None

        cb = bias_c + (-0.168736 * img[0, :, :] - 0.331264 * img[1, :, :] + 0.5 * img[2, :, :])
        cr = bias_c + (0.5 * img[0, :, :] - 0.418688 * img[1, :, :] - 0.081312 * img[2, :, :])

    else:
        bias_y = 16.*(max_val/255)
        bias_c = 128.*(max_val/255)

        y = bias_y + (65.481 * img[0, :, :] + 128.553 * img[1, :, :] + 24.966 * img[2, :, :]) / 255.

        if y_only:
            return y, None, None

        cb = bias_c + (-37.797 * img[0, :, :] - 74.203 * img[1, :, :] + 112.0 * img[2, :, :]) / 255.
        cr = bias_c + (112.0 * img[0, :, :] - 93.786 * img[1, :, :] - 18.214 * img[2, :, :]) / 255.

    return y, cb, cr


def ycbcr_to_rgb(img, max_val=1, im_type='png'):  # image always expected in C, H, W format
    """
    Converts YCbCr image to RGB.
    :param img: CxHxW Image.
    :param max_val: Image dynamic range.
    :param im_type: PNG or JPG format.
    :return:  Converted Image.
    """
    if im_type == 'jpg':
        bias = 128.*(max_val/255)

        r = img[0, :, :] + 1.402 * img[2, :, :] - 1.402 * bias
        g = img[0, :, :] - 0.344136 * img[1, :, :] - 0.714136 * img[2, :, :] + (0.714136 + 0.344136) * bias
        b = img[0, :, :] + 1.772 * img[1, :, :] - 1.772 * bias

    else:
        bias_r = 222.921*(max_val/255)
        bias_g = 135.576*(max_val/255)
        bias_b = 276.836*(max_val/255)

        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - bias_r
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + bias_g
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - bias_b

    return r, g, b


def ycbcr_convert(img, y_only=True, max_val=1, im_type='png', input='rgb'):
    """
    RGB to YCbCr converter using ITU-R BT.601 format (https://en.wikipedia.org/wiki/YCbCr).
    Can perform forward and inverse operation across different modes.
    :param img: Image to convert.  Must be in C, H, W format (channels, height, width)
    :param y_only: Select whether to output luminance channel only.
    :param max_val: Image maximum pixel value.
    :param im_type: Specify image type - different conversion performed for jpg images.
    :param input: Specify whether the input is in rgb or ycbcr format.
    :return: Transformed image.
    """

    if type(img) == np.ndarray:
        form = 'numpy'
    elif type(img) == torch.Tensor:
        form = 'torch'
    else:
        raise Exception('Unknown Type', type(img))

    if len(img.shape) == 4:
        img = img.squeeze(0)

    if input == 'ycbcr':
        a, b, c = ycbcr_to_rgb(img, max_val=max_val, im_type=im_type)
    elif input == 'rgb':
        a, b, c = rgb_to_ycbcr(img, max_val=max_val, y_only=y_only, im_type=im_type)

    if form == 'numpy':
        if y_only and input == 'rgb':
            return np.expand_dims(a, axis=0)
        else:
            return np.array([a, b, c])
    elif form == 'torch':
        if y_only and input == 'rgb':
            return torch.unsqueeze(a, 0)
        else:
            return torch.stack([a, b, c], 0)


class RGBtoYCbCrConverter:
    def __init__(self, im_type='jpg', y_only=True, max_val=1):
        """
        Class used by Pytorch Data handler to convert RGB images.
        :param im_type: PNG or JPG.
        :param y_only: Set as true to only retrieve luminance values.
        :param max_val: Image dynamic range.
        """
        self.im_type = im_type
        self.y_only = y_only
        self.max_val = max_val

    def __call__(self, image):
        return ycbcr_convert(image, y_only=self.y_only, max_val=self.max_val, im_type=self.im_type, input='rgb')

    def __repr__(self):
        return self.__class__.__name__ + '()'
###### ---- ######


###### Cropping ######
def center_crop(image, height, width):
    """
    Base center cropping function
    :param image: Input image.
    :param height: Image crop height.
    :param width: Image crop width.
    :return:
    """
    res_w = image.width - width
    res_h = image.height - height
    l_crop, top_crop = res_w//2, res_h//2
    return image.crop((l_crop, top_crop, width + l_crop, top_crop + height))


class CenterCrop:
    def __init__(self, height, width):
        """
        Class used for Pytorch center cropping.
        :param height: Image crop height.
        :param width: Image crop width.
        :param scale: Scale factor to enlarge from
        """
        self.height = height
        self.width = width

    def __call__(self, image):
        """
        Args:
            image (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        return center_crop(image, self.height, self.width)


def find_height_width_position_offset(img):
    """
    Finds offset w.r.t. 2D image to determine indices of picture height, width for an arbitrary number of channels;
    e.g. for 2D image with dims (C, H, W), height and width are at indices 1 and 2, respectively;
    for a 3D image with dims (C, D, H, W), height and width are at indices 2 and 3, respectively,
    i.e. offset of 1 w.r.t. 2D case

    :param img: Any image
    :return: offset to add to 2D case to find height/width positions
    """
    n_dims = img.dim()
    return n_dims - 3


def extract_image_patch(image, x, y, crop_size, offset_lr=0):

    if offset_lr == 0:
        crop = image[:, y:y + crop_size, x:x + crop_size]
    elif offset_lr == 1:
        crop = image[:, :, y:y + crop_size, x:x + crop_size]

    return crop


def entropy_patch_selection(image, crop_size, number_of_patches=1, selection='highest', max_pixel_val=1,
                            entropy_region_size=10):

    # image needs to be converted to grayscale (ycbcr), then changed to uint8 before entropy calculation
    entropy_array = entropy(np.uint8(ycbcr_convert(image, y_only=True, max_val=max_pixel_val,
                                                   im_type='jpg', input='rgb').squeeze() * (255 / max_pixel_val)),
                            rectangle(entropy_region_size, entropy_region_size))

    entropy_array = torch.nn.functional.avg_pool2d(
        torch.tensor(entropy_array).unsqueeze(0), kernel_size=crop_size, stride=1).squeeze().numpy()

    if number_of_patches == 1:
        x, y = np.unravel_index(entropy_array.argmax(), entropy_array.shape)
        return x, y
    else:
        x_idx = []
        y_idx = []

        for i in range(number_of_patches):
            # select highest/lowest entropy indices
            if selection == 'highest':
                x, y = np.unravel_index(np.nanargmax(entropy_array), entropy_array.shape)
            else:
                x, y = np.unravel_index(np.nanargmin(entropy_array), entropy_array.shape)

            entropy_array[max(0, x-crop_size):x+crop_size, max(0, y-crop_size):y+crop_size] = np.nan  # covering up all patches which touch with latest selected patch
            x_idx.append(x)
            y_idx.append(y)

        return x_idx, y_idx


def random_patch_selection(image, crop_size, offset_lr=0, number_of_patches=1):
    rnd_h = []
    rnd_w = []
    for i in range(number_of_patches):
        rnd_h.append(random.randint(0, max(0, image.size()[1+offset_lr] - crop_size)))
        rnd_w.append(random.randint(0, max(0, image.size()[2+offset_lr] - crop_size)))

    return rnd_h, rnd_w


def image_patch_selection(image_lr, crop_size, scale=1, image_hr=None, patch_type='random',
                          offset_lr=0, offset_hr=0, number_of_patches=1, predefined_patch_locations=None,
                          entropy_selection='highest', entropy_region_size=10):
    

    if len(image_lr.size()) == 4:
        image_lr = image_lr.squeeze()
        offset_lr = find_height_width_position_offset(image_lr)
    if image_hr is not None and image_hr != np.array(0) and len(image_hr.size()) == 4:
        image_hr = image_hr.squeeze()
        offset_hr = find_height_width_position_offset(image_hr)

    if patch_type == 'random':
        hs, ws = random_patch_selection(image_lr, crop_size, offset_lr, number_of_patches=number_of_patches)
    elif patch_type == 'entropy':
        hs, ws = entropy_patch_selection(image_lr, crop_size,
                                         number_of_patches=number_of_patches,
                                         selection=entropy_selection, entropy_region_size=entropy_region_size)
    elif patch_type == 'predefined':
        hs, ws = zip(*predefined_patch_locations)

    crops = []
    hr_crops = []
    for h, w in zip(hs, ws):
        crops.append(extract_image_patch(image_lr, w, h, crop_size))
        if image_hr is not None and image_hr != np.array(0):
            h_GT, w_GT = int(h * scale), int(w * scale)
            hr_crops.append(extract_image_patch(image_hr, w_GT, h_GT, int(crop_size*scale)))

    return crops, hr_crops, list(zip(hs, ws))
    ### TODO: Consider doing above more generalised (instead of specifying a case for every number of n_dims):
    # [UNTESTED!]
    # dim_W = n_dims
    # dim_H = n_dims - 1
    # # For Numpy arrays:
    # cropped_lr = image_lr.take(range(rnd_h, rnd_h + crop_size), dim=dim_H)
    # cropped_lr = cropped_lr.take(range(rnd_w, rnd_w + crop_size), dim=dim_W)
    # cropped_hr = image_hr.take(range(rnd_h_GT, rnd_h_GT + int(crop_size*scale)), dim=dim_H)
    # cropped_hr = cropped_hr.take(range(rnd_w_GT, rnd_w_GT + int(crop_size*scale)), dim=dim_W)
    #
    # For tensors:
    # cropped_lr = torch.index_select(image_lr, dim_H, torch.tensor(list(range(rnd_h, rnd_h + crop_size))))
    # cropped_lr = torch.index_select(cropped_lr, dim_W, torch.tensor(list(range(rnd_w, rnd_w + crop_size))))
    # cropped_hr = torch.index_select(image_hr, dim_H, torch.tensor(list(range(rnd_h_GT, rnd_h_GT + int(crop_size*scale)))))
    # cropped_hr = torch.index_select(cropped_hr, dim_W, torch.tensor(list(range(rnd_w_GT, rnd_w_GT + int(crop_size*scale)))))
###### ---- ######


###### Augmentations ######
def random_flip_rotate(*img, hflip=True, vflip=True, rot=True):
    # Modified from https://github.com/yuanjunchai/IKC/blob/2a846cf1194cd9bace08973d55ecd8fd3179fe48/codes/data/util.py
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        offset = find_height_width_position_offset(img)
        if hflip:
            img = torch.flip(img, [2 + offset])
        if vflip:
            img = torch.flip(img, [1 + offset])
        if rot90:
            img = torch.transpose(img, 1 + offset, 2 + offset)
        return img

    return [_augment(I) for I in img]

def colour_distortion(*img, dist_strength=1.0):
    # Modified from appendix of https://arxiv.org/pdf/2002.05709.pdf
    color_jitter = transforms.ColorJitter(0.8*dist_strength, 0.8*dist_strength, 0.8*dist_strength, 0.2*dist_strength)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])

    return [color_distort(I) for I in img]

###### Unused ######
def scale_and_luminance_crop(im, max_val=1, target_max=255):
    if type(im) == np.ndarray:
        im_np = np.copy(im)
    elif type(im) == torch.Tensor:
        im_np = im.numpy()
    else:
        raise Exception('Unknown Type', type(im))

    im_rgb = ycbcr_convert(im_np, input='ycbcr', max_val=max_val)
    im_rgb *= target_max/max_val
    im_rgb = np.clip(im_rgb, 0, target_max)
    im_ycbcr = ycbcr_convert(im_rgb, input='rgb', max_val=target_max, y_only=False)

    return im_ycbcr, im_rgb

# =============================================================================
# # Based on random_flip_rotate() above
# def random_flip_rotate_3d(*img, hflip=True, rot=True):
#     # Modified from https://github.com/yuanjunchai/IKC/blob/2a846cf1194cd9bace08973d55ecd8fd3179fe48/codes/data/util.py
#     hflip = hflip and random.random() < 0.5
#     vflip = rot and random.random() < 0.5
#     rot90 = rot and random.random() < 0.5
#
#     def _augment(img):
#         if hflip:
#             img = torch.flip(img, [3])
#         if vflip:
#             img = torch.flip(img, [2])
#         if rot90:
#             img = torch.transpose(img, 2, 3)
#         return img
#
#     return [_augment(I) for I in img]
# =============================================================================
###### ---- ######
