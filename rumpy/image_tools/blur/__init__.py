import os
import random

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import math

from rumpy.shared_framework.configuration import constants as sconst
from rumpy.sr_tools.helper_functions import normalize
from rumpy.image_tools.blur.bsrgan_utils import generate_kernel, bsrgan_blur
from rumpy.image_tools.blur.srmd_gaussian_blur import random_batch_kernel, PCA, SRMDPreprocessing, PCAEncoder
from rumpy.image_tools.blur.real_esrgan_blur import random_mixed_kernels, select_specific_kernel, filter2D
from rumpy.shared_framework.configuration.gpu_check import device_selector


def read_pca_matrix(name, device):
    """Read standard PCA matrix from those defined in configuration file"""
    base_path = os.path.join(sconst.code_base_directory, 'shared_framework', 'configuration')

    if name == 'standard_10_component':
        pca_matrix_path = os.path.join(base_path, 'standard_blur_10_component_pca_matrix.pth')
    elif name == 'extended_blur_100_component':
        pca_matrix_path = os.path.join(base_path, 'extended_blur_100_component_pca_matrix.pth')
    elif 'pca_matrix.pth' in name:
        pca_matrix_path = name
    else:
        raise RuntimeError('PCA matrix identity not recognized')

    if isinstance(device, int):  # fix for GPUs
        dev_loc = "cuda:%d" % device
    else:
        dev_loc = device

    return torch.load(pca_matrix_path, map_location=dev_loc).float()


class Blur:
    """
    Basic class containing machinery for generating and managing blur kernel degradations.
    """

    def __init__(self, device, kernel_size, request_full_kernels, normalize_metadata,
                 request_pca_kernels, load_pca_matrix, pca_batch_len, pca_length, request_kernel_metadata):
        """
        :param device: GPU or CPU device (GPU provides speed-ups).
        :param kernel_size: Set kernel size (needs to be an odd number).
        :param request_full_kernels: Set to true to request full blur kernel output.
        :param normalize_metadata: Set to true to ensure metadata output is between 0 and 1 (currently not in use).
        :param request_pca_kernels: Request a PCA-ed version of the kernel used for each image.
        :param load_pca_matrix: Use a pre-made matrix for PCA.
        :param pca_batch_len: Quantity of kernels to generate before producing a PCA representation.
        :param pca_length: PCA output encoding length.
        :param request_kernel_metadata: Set to true to request blur kernel parameters as metadata output.
        """

        self.request_pca_kernels = request_pca_kernels
        self.request_full_kernels = request_full_kernels
        self.kernel_size = kernel_size
        self.normalize_metadata = normalize_metadata
        self.request_kernel_metadata = request_kernel_metadata

        self.device = device_selector(True, device)

        if self.request_pca_kernels:
            if load_pca_matrix:
                pca_matrix = read_pca_matrix(load_pca_matrix, self.device)
            else:
                print('Generating PCA encoding for %s system' % self.__class__.__name__)
                # generates pca matrix through repeated generation of different kernels
                kernel_array = self.generate_random_kernels(pca_batch_len)

                pca_matrix = PCA(kernel_array, k=pca_length).float()

            self.pca_encoder = PCAEncoder(pca_matrix, device=self.device)

        if self.device != torch.device('cpu') and torch.multiprocessing.get_start_method() != 'spawn':
            torch.multiprocessing.set_start_method('spawn', force=True)
            print('Multiprocessing start method changed to:', torch.multiprocessing.get_start_method())

    def generate_random_kernels(self, batch_len):
        """
        Generate a batch of kernels given the batch size and using the parameters from the class initialisation.

        :param batch_len: The batch size for the batch of kernels to be generated.
        """
        raise NotImplementedError('Needs to be implemented for each individual blurring system.')

    def get_hyperparams(self):
        """
        Get extra information on the blurring parameters and the system used.
        """
        raise NotImplementedError('Needs to be implemented for each individual blurring system.')

    def save_pca_matrix(self, location):
        """
        Save the PCA matrix if it has been generated.

        :param location: Location to save the PCA matrix file (.pth).
        """
        if self.request_pca_kernels:
            torch.save(self.pca_encoder.weight, os.path.join(location, '%s_pca_matrix.pth' % self.__class__.__name__))
        else:
            print('No PCA kernels saved, as these have not been generated.')


class RealESRGANBlur(Blur):
    """
    Blurring system used by Real-ESRGAN.
    Provides:
    - Isotropic Gaussian kernels (iso)
    - Anisotropic Gaussian kernels (aniso)
    - Isotropic generalized Gaussian kernels (generalized_iso)
    - Anisotropic generalized Gaussian kernels (generalized_aniso)
    - Plateau-like isotropic kernels (plateau_iso)
    - Plateau-like anisotropic kernels (plateau_aniso)
    - 2D Sinc filter kernels

    Kernel convolution can be sped up using a GPU.
    """

    def __init__(self,
                 # generic params
                 request_pca_kernels=False, pca_length=10, pca_batch_len=30000, load_pca_matrix=None,
                 normalize_metadata=True, request_full_kernels=False, kernel_size=21, device=torch.device('cpu'),
                 request_kernel_metadata=False,
                 # specific params
                 kernel_range=('iso',), kernel_probabilities=None, semi_random_selection=False,
                 sigma_x_range=(0.6, 5), sigma_y_range=(0.6, 5),
                 rotation_range=(-math.pi, math.pi), betag_range=(0.5, 8),
                 betap_range=(0.5, 8), noise_range=None,
                 random_selection=True, selected_kernel=None, use_kernel_code=False, **specific_blur_params
                 ):
        """
        Generic parameter descriptions are described in the Blur class.

        :param random_selection: Set to true to select a kernel randomly for each invocation.
        :param noise_range:  Specify this noise range to add extra noise to Gaussian kernels after generation.
        :param use_kernel_code: Set this flag to convert kernel type string to standard kernel code in metadata output.

        Random Kernels
        :param kernel_range: Which types of kernels to sample (list).
        :param kernel_probabilities: Probability assigned to each kernel type.
        :param semi_random_selection: Set this to true to use a mix of specified and randomised blur params
        :param sigma_x_range: Sigma x (width) range (for anisotropic kernels)
        :param sigma_y_range: Sigma y (width) range (for anisotropic kernels)
        :param rotation_range: Rotation range (for anisotropic kernels)
        :param betag_range:  plateau-like anisotropic kernel shape range
        :param betap_range:  plateau-like anisotropic kernel shape range

        Specific Kernels
        :param selected_kernel: Specific selected kernel type.
        Specific kernel parameters can then be set by setting one of the random variables without 'range' e.g. set
        sigma_x by setting the value of 'sigma_x'.
        """

        if random_selection and semi_random_selection:
            raise RuntimeError('Both random and semi random modes cannot be on simultaneously.')
        self.random_selection = random_selection
        self.semi_random_selection = semi_random_selection
        if not random_selection and not semi_random_selection:
            # will only use the specified kernel
            if selected_kernel is None:
                raise RuntimeError('Need to specify requested kernel if not using random selection.')
            self.selected_kernel = selected_kernel
            self.kernel_probabilities = None
            self.kernel_type_range = None
        else:
            # will randomly select kernel from provided range
            if kernel_range == 'all':
                self.kernel_type_range = ['iso', 'aniso', 'generalized_iso',
                                          'generalized_aniso', 'plateau_iso',
                                          'plateau_aniso', 'sinc']
            else:
                self.kernel_type_range = kernel_range  # valid types of kernels

            self.kernel_probabilities = kernel_probabilities  # probability to select each one
            self.selected_kernel = None

        # degradation parameters
        self.kernel_params = specific_blur_params
        self.sigma_x_range = sigma_x_range  # TODO: is there a way to shorten all this code?
        self.sigma_y_range = sigma_y_range
        self.rotation_range = rotation_range
        self.betag_range = betag_range
        self.betap_range = betap_range
        self.noise_range = noise_range
        self.use_kernel_code = use_kernel_code

        super().__init__(device, kernel_size, request_full_kernels, normalize_metadata,
                         request_pca_kernels, load_pca_matrix, pca_batch_len, pca_length, request_kernel_metadata)

    def generate_single_kernel(self):
        """
        Generate a single blur kernel using the parameters defined in the class initialisation.
        """

        if self.random_selection:
            kernel, metadata = random_mixed_kernels(self.kernel_type_range,
                                                    self.kernel_probabilities, self.kernel_size,
                                                    self.sigma_x_range, self.sigma_y_range, self.rotation_range,
                                                    self.betag_range, self.betap_range,
                                                    self.noise_range)

            # standardizes metadata for csv file output
            for mp in ['sigma_x', 'sigma_y', 'rotation', 'beta_p', 'beta_g', 'omega_c']:
                if mp not in metadata:
                    metadata[mp] = 0

        elif self.semi_random_selection:
            # in this mode, some params can be fixed and others randomised
            selected_params = {}
            for key, val in self.kernel_params.items():  # extracts fixed params
                selected_params[key] = val
            kernel_type = random.choices(self.kernel_type_range, self.kernel_probabilities)[0]  # randomly selects kernel type

            for param, range in zip(['sigma_x', 'sigma_y', 'rotation', 'beta_p', 'beta_g'], # assigns random value to params which weren't specified
                                    [self.sigma_x_range, self.sigma_y_range, self.rotation_range, self.betap_range,
                                     self.betag_range]):
                if param not in selected_params:
                    selected_val = np.random.uniform(range[0], range[1])
                    selected_params[param] = selected_val

            selected_params['omega_c'] = np.random.uniform(np.pi / 3, np.pi)

            kernel = select_specific_kernel(
                kernel_type,
                self.kernel_size,
                **selected_params,
                noise_range=self.noise_range)

            metadata = {**selected_params, **{'kernel_type': kernel_type}}

        else:
            kernel = select_specific_kernel(
                self.selected_kernel,
                self.kernel_size,
                **self.kernel_params,
                noise_range=self.noise_range)

            metadata = {**self.kernel_params, **{'kernel_type': self.selected_kernel}}

        metadata['kernel_size'] = self.kernel_size

        if self.use_kernel_code:  # converts text to digit code
            metadata['kernel_type'] = sconst.blur_kernel_code_conversion[metadata['kernel_type']]

        return kernel, metadata

    def generate_random_kernels(self, batch_len):
        if not self.random_selection:
            raise RuntimeError('Cannot generate random kernels as system set up for deterministic blurring.')

        kernel_array = np.zeros((batch_len, self.kernel_size, self.kernel_size))

        for i in tqdm(range(batch_len)):
            kernel, _ = self.generate_single_kernel()
            kernel_array[i, ...] = kernel

        kernel_array = kernel_array.reshape((batch_len, -1))

        return kernel_array

    def get_hyperparams(self):
        params = {'blur_type': 'real_esrgan',
                  'kernel_size': self.kernel_size,
                  'sigma_x_range': self.sigma_x_range,
                  # ranges are still exported, as they can be used for normalization
                  'sigma_y_range': self.sigma_y_range,
                  'rotation_range': self.rotation_range,
                  'beta_p_range': self.betap_range,
                  'beta_g_range': self.betag_range,
                  'noise_range': self.noise_range}
        if self.random_selection:
            params = {**params, **{
                'kernel_type_range': self.kernel_type_range,
                'kernel_probabilities': self.kernel_probabilities,
                'sigma_x_range': self.sigma_x_range,
                'sigma_y_range': self.sigma_y_range,
                'rotation_range': self.rotation_range,
                'beta_g_range': self.betag_range,
                'beta_p_range': self.betap_range
            }}
        else:
            params = {**params, **self.kernel_params, **{'kernel_type': self.selected_kernel}}
        return params

    def __call__(self, image):  # TODO: how can I batch kernels together to get a speed-up?
        kernel, metadata = self.generate_single_kernel()
        torch_kernel = torch.FloatTensor(kernel).unsqueeze(0).to(self.device)
        tensor_image = filter2D(transforms.ToTensor()(image).unsqueeze(0).to(self.device), torch_kernel)

        pil_image = transforms.ToPILImage()(tensor_image.squeeze(0).cpu())

        meta_dict = {}
        if self.request_full_kernels:
            meta_dict['unmodified_blur_kernel'] = kernel.reshape(1, -1).squeeze().tolist()
        if self.request_pca_kernels:
            pca_kernel = self.pca_encoder(torch_kernel.reshape(1, -1)).tolist()[0]
            meta_dict['blur_kernel'] = pca_kernel
        if self.request_kernel_metadata:
            if self.normalize_metadata:
                metadata['sigma_x'] = normalize(metadata['sigma_x'], self.sigma_x_range[0], self.sigma_x_range[1])
                metadata['sigma_y'] = normalize(metadata['sigma_y'], self.sigma_y_range[0], self.sigma_y_range[1])

            meta_dict = {**metadata, **meta_dict}

        return pil_image, meta_dict


class SRMDGaussianBlur(Blur):
    """
    Blurring system used by SRMD and IKC.  Provides isotropic and anisotropic Gaussian kernels.
    Can be used with GPU for processing speed-up.  Random kernels by default off!

    Noise system not in use.
    """

    def __init__(self,
                 # general params
                 request_pca_kernels=False, pca_length=10, pca_batch_len=30000, load_pca_matrix=None,
                 normalize_metadata=True, request_full_kernels=False, kernel_size=21, device=torch.device('cpu'),
                 request_kernel_metadata=False,
                 # specific params
                 random=False, noise=False, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, rate_cln=0.2,
                 noise_high=0.08):
        """
        Generic parameter descriptions are described in the Blur class.

        :param request_kernel_metadata:
        :param random: Set to true to generate randomly-sized blur kernels for each input
        :param noise: Set to true to include noise addition after blurring
        :param sig: Blur kernel sigma to use (if not random)
        :param sig_min: Minimum value of blur kernel sigma to use (if random)
        :param sig_max: Maximum value of blur kernel sigma to use (if random)
        :param rate_iso: Probability of generating an isotropic kernel (set to 1.0 to never generate anisotropic kernels)
        :param scaling:  Maximum ratio of anisotropic dimension scale (actual scale will be randomly selected)
        :param rate_cln: TODO: investigate noise mechanism
        :param noise_high: TODO: investigate noise mechanism
        """

        super().__init__(device, kernel_size, request_full_kernels, normalize_metadata,
                         request_pca_kernels, load_pca_matrix, pca_batch_len, pca_length, request_kernel_metadata)

        self.blur_mechanism = SRMDPreprocessing(device=device,
                                                random=random, para_input=pca_length, kernel=kernel_size, noise=noise,
                                                sig=sig, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso,
                                                scaling=scaling, rate_cln=rate_cln, noise_high=noise_high)
        # TODO: do unmodified blur kernels need normalization?

    def generate_random_kernels(self, batch_len):
        batch_ker, _ = random_batch_kernel(batch=batch_len, tensor=False, kernel=self.kernel_size)
        b = np.size(batch_ker, 0)
        batch_ker = batch_ker.reshape((b, -1))

        return batch_ker

    def get_hyperparams(self):
        sig_params = {}
        if self.blur_mechanism.random:
            sig_params['random'] = 'True'
            sig_params['max_sigma'] = self.blur_mechanism.kernel_gen.sig_max
            sig_params['min_sigma'] = self.blur_mechanism.kernel_gen.sig_min
        else:
            sig_params['random'] = 'False'
            sig_params['sigma'] = self.blur_mechanism.kernel_gen.sig

        const_params = {
            'blur_type': 'srmd',
            'kernel_size': self.blur_mechanism.kernel_gen.l,
            'isotropic_probability': self.blur_mechanism.kernel_gen.rate,
            'anisotropic_scaling': self.blur_mechanism.kernel_gen.scaling,
        }

        return {**sig_params, **const_params}

    def postprocess_metadata(self, metadata):
        """
        Adjust any metadata values after they have been generated.

        :param metadata: Value of parameters or list of parameters from the generated metadata.
        :return: Dictionary containing adjusted/filled-in metadata parameters.
        """
        if self.blur_mechanism.kernel_gen.rate == 1.0:
            return {'isotropic_sigma': metadata}
        else:
            if isinstance(metadata, tuple):
                return {'isotropic_sigma': 0,
                        'anisotropic_x': metadata[1],
                        'anisotropic_y': metadata[2],
                        'anisotropic_pi': metadata[0]}
            else:
                return {'isotropic_sigma': metadata,
                        'anisotropic_x': 0,
                        'anisotropic_y': 0,
                        'anisotropic_pi': 0}

    def __call__(self, image):
        tensor_image, unreduced_kernel, metadata = self.blur_mechanism(transforms.ToTensor()(image))
        blurred_image = transforms.ToPILImage()(tensor_image.squeeze(0).cpu())

        # TODO: include the following if including noise mechanism:
        # re_code = torch.cat([kernel_code, Noise_level * 10], dim=1) if self.noise else kernel_code

        blur_kernel = unreduced_kernel.cpu().numpy().squeeze().flatten().tolist()

        meta_dict = {}
        if self.request_full_kernels:
            meta_dict['unmodified_blur_kernel'] = blur_kernel
        if self.request_pca_kernels:
            # kernel encode (via PCA)
            kernel_code = self.pca_encoder(unreduced_kernel.reshape(1, -1))  # B x self.para_input
            pca_kernel = kernel_code.cpu().tolist()[0]
            meta_dict['blur_kernel'] = pca_kernel
        if self.request_kernel_metadata:
            meta_dict = {**self.postprocess_metadata(metadata[0]), **meta_dict}

        return blurred_image, meta_dict


class BSRGANBlur(Blur):
    """
    Main blurring system used by BSRGAN.  Provides Gaussian isotropic and anisotropic blurring.
    Current implementation very slow, not recommended for online degradation.
    TODO: needs optimization and code needs to be made more concise, plus some features not available
    """

    def __init__(self,
                 # general params
                 request_pca_kernels=False, pca_length=10, pca_batch_len=30000, load_pca_matrix=None,
                 normalize_metadata=True, request_full_kernels=False, kernel_size=21, device=torch.device('cpu'),
                 request_kernel_metadata=False,
                 # specific params
                 l1_range=(0.0, 8.0), l2_range=(0.0, 8.0), sigma_range=(0.1, 2.8),
                 theta_range=(0, np.pi)):  # TODO: add definition for specific values, not just ranges

        self.l1_range = l1_range
        self.l2_range = l2_range
        self.theta_range = theta_range
        self.sigma_range = sigma_range

        super().__init__(device, kernel_size, request_full_kernels, normalize_metadata,
                         request_pca_kernels, load_pca_matrix, pca_batch_len, pca_length, request_kernel_metadata)

    def generate_random_kernels(self, batch_len):
        kernel_array = np.zeros((batch_len, self.kernel_size, self.kernel_size))
        for i in tqdm(range(batch_len)):  # Process very slow!
            sig = self.generate_random_float_in_range(self.sigma_range)
            l1 = self.generate_random_float_in_range(self.l1_range)
            l2 = self.generate_random_float_in_range(self.l2_range)
            theta = self.generate_random_float_in_range(self.theta_range)
            k = generate_kernel(sigma=sig, kernel_size=self.kernel_size, l1=l1, l2=l2, theta=theta, gauss_type='random')
            kernel_array[i, ...] = k

        kernel_array = kernel_array.reshape((batch_len, -1))
        return kernel_array

    def generate_random_float_in_range(self, num_range):
        """
        Generate a random floating point number in given a range.

        :param num_range: Tuple containing the min and max for the random number.
        :return: Random floating point number.
        """
        output_num = (random.random() * (num_range[1] - num_range[0])) + num_range[0]

        return output_num

    def get_hyperparams(self):
        return {'blur_type': 'bsrgan',
                'min_sigma': self.sigma_range[0],
                'max_sigma': self.sigma_range[1],
                'min_l1': self.l1_range[0],
                'max_l1': self.l1_range[1],
                'min_l2': self.l2_range[0],
                'max_l2': self.l2_range[1],
                'min_theta': self.theta_range[0],
                'max_theta': self.theta_range[1]
                }

    def __call__(self, image):
        sig = self.generate_random_float_in_range(self.sigma_range)
        l1 = self.generate_random_float_in_range(self.l1_range)
        l2 = self.generate_random_float_in_range(self.l2_range)
        theta = self.generate_random_float_in_range(self.theta_range)

        blurred_image, blur_kernel = bsrgan_blur(image, sigma=sig, l1=l1, l2=l2, theta=theta, gauss_type='random')
        blurred_image = transforms.ToPILImage()(blurred_image)

        meta_dict = {}
        if self.request_full_kernels:  # TODO: confirm unmodified blur kernel is exporting correctly
            meta_dict['unmodified_blur_kernel'] = blur_kernel
        else:
            pca_kernel = \
                self.pca_encoder(torch.from_numpy(blur_kernel.astype('float32')).unsqueeze(0).reshape(1, -1)).tolist()[
                    0]
            meta_dict['blur_kernel'] = pca_kernel

        return blurred_image, meta_dict
