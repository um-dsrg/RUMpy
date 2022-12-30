import random
import numpy as np
from torchvision import transforms
import torch

from rumpy.sr_tools.helper_functions import normalize
from .real_esrgan_noise import random_add_poisson_noise_pt, random_add_gaussian_noise_pt, \
    generate_gaussian_noise_pt, generate_poisson_noise_pt
from rumpy.image_tools.blur.srmd_gaussian_blur import PCA, PCAEncoder
from rumpy.shared_framework.configuration.gpu_check import device_selector
from ..blur import read_pca_matrix


class RealESRGANNoise:
    """
    Additive noise system used by Real-ESRGAN.
    Provides Gaussian/Poisson Color or Gray noise.
    """

    def __init__(self, normalize_metadata=True, gaussian_poisson_ratio=0.5,
                 poisson_noise_scale_range=(0, 1.0),
                 gaussian_noise_sigma_range=(0, 1.0),
                 request_noise_image_pca=False,
                 noise_image_pca_length=100,
                 pca_patch_size=64,
                 pca_batch_len=500,
                 gray_noise_probability=0.4,
                 load_pca_matrix=None,
                 random_noise_generation=True,
                 device=torch.device('cpu'),
                 **specific_noise_params):
        """
        Noise generator used by Real-ESRGAN
        :param normalize_metadata: Set to true to normalize all metadata outputs to 0, 1 (based on provided ranges)
        :param gaussian_poisson_ratio: Probabilty ratio between Poisson noise and Gaussian noise
        :param poisson_noise_scale_range: Range of scales to apply when using Poisson noise (minimum, maximum)
        :param gaussian_noise_sigma_range: Range of sigmas to apply when using Gaussian noise (minimum, maximum)
        :param random_noise_generation: Set to False to select specific noise to generate.
        Set values of this noise by setting 'poisson_noise_scale', 'gaussian_noise_scale' and 'gray_noise'.
        If 'gray_noise' is not set, it will be randomly selected using gray_noise_probability.
        If both poisson and gaussian noise scales are set, the type will be randomly  selected using gaussian_poisson_ratio.
        :param device: Specify a GPU device here if computations should be done on GPU to speed-up processing

        PCA params (not fully tested)
        :param request_noise_image_pca: Set to true to also output the PCA-ed noise image with metadata
        :param noise_image_pca_length: Length of PCA output for noise image
        :param pca_batch_len: Number of noise examples to generate to train PCA encoder
        :param pca_patch_size: Noise image patch size to use when training PCA encoder
        :param gray_noise_probability: Probability to generate grey noise versus colour noise (multichannel)
        :param load_pca_matrix: Location of pre-defined PCA matrix
        """

        self.gaussian_poisson_ratio = gaussian_poisson_ratio
        self.poisson_noise_scale_range = poisson_noise_scale_range
        self.gaussian_noise_sigma_range = gaussian_noise_sigma_range
        self.gray_noise_probability = gray_noise_probability
        self.random_noise = random_noise_generation
        self.specific_noise_params = specific_noise_params

        self.device = device_selector(True, device)

        self.request_noise_image_pca = request_noise_image_pca

        self.normalize_metadata = normalize_metadata

        if isinstance(self.device, str) and 'cuda' in self.device and torch.multiprocessing.get_start_method() != 'spawn':
            torch.multiprocessing.set_start_method('spawn', force=True)
            print('Multiprocessing start method changed to:', torch.multiprocessing.get_start_method())

        if request_noise_image_pca:
            # This PCA system should theoretically be able to represent a small noise patch with a lesser number of dimensions.
            # However, Poisson noise is signal-dependent, and is thus not easy to represent with a PCA system.
            # The machinery to PCA a noise patch is still in place here, but will not reproducibly model the noise correctly.
            if load_pca_matrix:
                pca_matrix = read_pca_matrix(load_pca_matrix, self.device)
            else:
                # provides enough data examples for PCA to train through repeated generation of different noise examples
                blank_image_gauss = torch.zeros((int(pca_batch_len/2), 3, pca_patch_size, pca_patch_size))
                blank_image_poisson = torch.zeros((int(pca_batch_len/2), 3, pca_patch_size, pca_patch_size))

                print('Generating PCA encoding for Real-ESRGAN noise system')

                _, _, noise_image_gauss = random_add_gaussian_noise_pt(blank_image_gauss,
                                                                       sigma_range=self.gaussian_noise_sigma_range,
                                                                       gray_prob=self.gray_noise_probability)

                _, _, noise_image_poisson = random_add_poisson_noise_pt(blank_image_poisson,
                                                                        scale_range=self.poisson_noise_scale_range,
                                                                        gray_prob=self.gray_noise_probability)
                full_noise_array = np.concatenate((noise_image_gauss, noise_image_poisson))

                full_noise_array = full_noise_array.reshape((pca_batch_len, -1))
                pca_matrix = PCA(full_noise_array, k=noise_image_pca_length).float()

            self.pca_encoder = PCAEncoder(pca_matrix, device=self.device)
            self.pca_cropper = transforms.CenterCrop(pca_patch_size)
        else:
            self.pca_encoder = None
            self.pca_cropper = None

    def get_hyperparams(self):
        """
        Get extra information on the noise parameters and the probabilities used.
        """
        return {
            'gaussian_poisson_ratio': self.gaussian_poisson_ratio,
            'poisson_noise_scale_range': self.poisson_noise_scale_range,
            'gaussian_noise_sigma_range': self.gaussian_noise_sigma_range,
            'gray_noise_probability': self.gray_noise_probability
        }

    def __call__(self, image):
        # NOTE: Old system used ToTensor but couldn't set the device, so the conversion to tensor is done differently
        # The permute changes the position of the channel dimension
        torch_im = torch.tensor(np.asarray(image) / 255.0, device=self.device).permute(2, 0, 1).unsqueeze(0)

        if self.random_noise:
            if random.random() < self.gaussian_poisson_ratio:
                noise_augmented_image, metadata, noise_image = random_add_gaussian_noise_pt(torch_im,
                                                                                            sigma_range=self.gaussian_noise_sigma_range,
                                                                                            gray_prob=self.gray_noise_probability)
                n_type = 'gaussian'
            else:
                noise_augmented_image, metadata, noise_image = random_add_poisson_noise_pt(torch_im,
                                                                                           scale_range=self.poisson_noise_scale_range,
                                                                                           gray_prob=self.gray_noise_probability)
                n_type = 'poisson'
        else:

            # if both types of noise scales are given, one is chosen randomly
            if self.specific_noise_params['gaussian_noise_scale'] > 0 and self.specific_noise_params['poisson_noise_scale'] > 0:
                if random.random() < self.gaussian_poisson_ratio:
                    n_type = 'gaussian'
                else:
                    n_type = 'poisson'
            else:
                if self.specific_noise_params['gaussian_noise_scale'] > 0.0:
                    n_type = 'gaussian'
                else:
                    n_type = 'poisson'

            if 'gray_noise' not in self.specific_noise_params:  # if gray_noise is not given, this is randomly determined
                if random.random() < self.gray_noise_probability:
                    gray_noise = 0.0
                else:
                    gray_noise = 1.0
            else:
                gray_noise = self.specific_noise_params['gray_noise']

            if gray_noise != 0 and gray_noise != 1:
                raise RuntimeError('gray noise must be 1 or 0, not in between.')

            if n_type == 'gaussian':
                gauss_scale = self.specific_noise_params['gaussian_noise_scale']
                metadata = {'gaussian_noise_scale': gauss_scale,
                            'gray_noise': gray_noise,
                            'poisson_noise_scale': 0.0}
                noise_image = generate_gaussian_noise_pt(torch_im,
                                                         gauss_scale,
                                                         gray_noise)
            else:
                poiss_scale = self.specific_noise_params['poisson_noise_scale']
                metadata = {'gaussian_noise_scale': 0.0,
                            'gray_noise': gray_noise,
                            'poisson_noise_scale': poiss_scale}
                noise_image = generate_poisson_noise_pt(torch_im, poiss_scale, gray_noise)

            noise_augmented_image = torch.clamp(torch_im + noise_image, 0, 1)

        pil_image = transforms.ToPILImage()(noise_augmented_image.squeeze(0).cpu())

        # Trying to avoid running out of memory
        del torch_im
        del noise_augmented_image
        del noise_image

        if isinstance(self.device, str) and 'cuda' in self.device:  # should only delete unused tensors not useful data
            torch.cuda.empty_cache()

        if self.normalize_metadata:
            if n_type == 'poisson':
                metadata['poisson_noise_scale'] = normalize(metadata['poisson_noise_scale'],
                                                            self.poisson_noise_scale_range[0],
                                                            self.poisson_noise_scale_range[1])
            else:
                metadata['gaussian_noise_scale'] = normalize(metadata['gaussian_noise_scale'],
                                                             self.gaussian_noise_sigma_range[0],
                                                             self.gaussian_noise_sigma_range[1])

        if self.request_noise_image_pca:
            noise_crop = self.pca_cropper(noise_image)  # TODO: this would have been deleted earlier - need to fix!
            noise_pca = self.pca_encoder(noise_crop.reshape((1, -1))).tolist()[0]
            metadata['pca_noise'] = noise_pca
        return pil_image, metadata
