import os
import re
import numpy as np
import PIL.Image
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pandas as pd
import toml
import shutil
from colorama import init, Fore
from pydoc import locate

from rumpy.shared_framework.configuration import constants as sconst
import rumpy.image_tools.blur.srmd_gaussian_blur as g_utils
from rumpy.sr_tools.helper_functions import extract_image_names_from_folder, create_dir_if_empty, convert_default_none_dict
from rumpy.image_tools import available_tools
from rumpy.image_tools.compression import JMCompress

init()  # colorama setup


def remove_file_or_folder(item_path):
    if os.path.isdir(item_path):
        shutil.rmtree(item_path)
    elif os.path.isfile(item_path):
        os.remove(item_path)


def clean_scratch_dir(specific_dir=None):

    if specific_dir is None:
        scratch_contents = os.listdir(sconst.scratch_directory)

        for item in scratch_contents:  # deleting directory when done with temp JM files
            match = re.search(r'(\d+h-\d+m-\d+s-\w+-\d+-\d+)', item)

            if match is not None:
                item_path = os.path.join(sconst.scratch_directory, item)
                remove_file_or_folder(item_path)
    else:
        item_path = os.path.join(sconst.scratch_directory, specific_dir)
        remove_file_or_folder(item_path)


def read_image(filename):
    """Function for easy reading of an image using PIL"""
    im = PIL.Image.open(filename)
    # just in case images have transparency channel included, or are grayscale
    if im.mode == 'RGBA' or im.mode == 'L':
        im = im.convert('RGB')
    return im


class ImagePipeline:
    """
    Main class taking care of converting images with a set of pre-defined image transformation functions.
    """
    def __init__(self, pipeline, deg_configs=None, output_extension='.png', **kwargs):
        """
        :param pipeline: string of commands and corresponding configuration names containing details on
        functions to be applied on images.
        :param deg_configs: Dictionary of configurations, with keynames matching those in the pipeline.
        :param output_extension: Output extension to use for all newly-generated images.
        :param kwargs: N/A
        # TODO: add an example
        """

        if all(isinstance(i, list) for i in pipeline):  # TODO: deal with case where default is not specified for all operations
            operations, configs = zip(*pipeline)  # full config pipeline
        else:
            operations = pipeline  # all defaults pipeline
            configs = ['default'] * len(operations)

        operations = [op.lower() for op in operations]  # convert to lowerscale for dict compatibility

        self.pipeline = OrderedDict()  # TODO: deal with double zip...

        if 'jmcompress' in operations or 'randomcompress' in operations:
            self.jm_present = True
            self.jm_cleanup_files = []
        else:
            self.jm_present = False

        self.blur_present = None

        for index, (operation, config) in enumerate(zip(operations, configs)):
            if config == 'default':
                op_params = {}
            else:
                op_params = {k: v for k, v in deg_configs[config].items()}  # converts defaultdict to normal dict.  Not sure why this is required to be able to run parallel dataloaders.

            if operation == 'downsample':
                if 'scale' in kwargs:
                    op_params['scale'] = kwargs['scale']

            if operation == 'downsample' and self.jm_present:
                op_params['jm'] = True
                print('%sSince JM compression is part of the provided pipeline, '
                      'the downscaling operation will ensure output is JM-compliant.%s' % (Fore.YELLOW, Fore.RESET))

            self.pipeline[(index, operation)] = self.define_degradation(operation, **op_params)

            if operation == 'jmcompress':
                self.jm_cleanup_files.extend(self.pipeline[(index, operation)].temp_files)

            if operation == 'randomcompress':
                self.jm_cleanup_files.extend(self.pipeline[(index, operation)].jm_class.temp_files)

            if operation == 'srmdgaussianblur' or operation == 'bsrganblur' or operation == 'realesrganblur':
                self.blur_present = (index, operation)  # mark which operation contains blurring

        # TODO: define setup of random operation selection

        self.output_extension = output_extension

    @staticmethod
    def define_degradation(name, **kwargs):
        """Locates and loads degradation from available set of degradation classes"""
        return locate(available_tools[name])(**kwargs)

    def _format_metadata(self, metadata, step, operation):
        """
        Standard formatting for metadata information.

        :param metadata: Dictionary of metadata names/values.
        :param step: Order of operation used.
        :param operation: Operation name.
        :return: Formatted metadata dictionary
        """
        new_metadata = {}
        for attribute, value in metadata.items():
            new_metadata['%s-%s-%s' % (step, operation, attribute)] = value
        return new_metadata

    def run_pipeline(self, images=None, image_files=None, save_to_dir=None, progress_bar_off=False, multiples=1):
        """
        Main process for applying supplied set of transformations to images.
        Images can either be supplied directly or as a file location for on-line reading.
        :param images: List of images or single image.  CxHxW format.
        :param image_files: List of image file locations or single file location (str).
        :param save_to_dir: Save processed files to specified directory.
        :param progress_bar_off:  Set to true to turn off progress bar (useful when multiprocessing).
        :param multiples: Number of copies to generate from each image.
        :return: processed images, any metadata generated, metadata keys/names
        """

        # TODO: also accept image dictionaries?
        if (images is None and image_files is None) or (images is not None and image_files is not None):
            raise RuntimeError('Either image variables or image files need to be provided.')

        if isinstance(image_files, str):
            image_files = [image_files]

        # variable setup
        pipeline_images = OrderedDict()
        final_images = []
        final_metadata = OrderedDict()

        if image_files is None:  # temporary name provided for unnamed images
            if type(images) != list:
                images = [images]
            for index, image in enumerate(images):
                pipeline_images['temp_name_%d' % index] = image
        else:
            for image in image_files:  # uses image basename for key if files provided
                b_name = os.path.splitext(os.path.basename(image))[0] + self.output_extension
                pipeline_images[b_name] = image

        if save_to_dir and self.blur_present:  # saves blur kernel PCA encoding matrix
            self.pipeline[self.blur_present].save_pca_matrix(save_to_dir)

        # Pipeline application
        if not progress_bar_off:
            loader = tqdm(pipeline_images.items())
        else:
            loader = pipeline_images.items()

        for index, (image_name, image) in enumerate(loader):
            if isinstance(image, str):  # TODO: deal with situation where multiple images are spawned from one image
                flux_im = read_image(image)
            else:
                flux_im = image

            start_im = flux_im.copy()

            for m in range(multiples):
                flux_im = start_im.copy()

                metadata_dict = {}
                for key, operation in self.pipeline.items():  # Running all functions in provided pipeline
                    flux_im, metadata = operation(flux_im)
                    metadata = self._format_metadata(metadata, key[0], key[1])
                    metadata_dict = {**metadata_dict, **metadata}

                if multiples == 1:
                    lr_image_name = image_name
                else:
                    dot = image_name.find('.')
                    lr_image_name = image_name[:dot] + '_q' + str(m) + image_name[dot:]

                final_metadata[lr_image_name] = metadata_dict

                if save_to_dir:
                    flux_im.save(os.path.join(save_to_dir, lr_image_name))
                else:
                    final_images.append(flux_im)  # TODO: is a new dictionary necessary?

        # metadata saving
        if save_to_dir:
            # saving all pipeline metadata for each image
            saveable_metadata = pd.DataFrame.from_dict(final_metadata, orient='index')
            saveable_metadata.index.rename('image', inplace=True)
            saveable_metadata.to_csv(os.path.join(save_to_dir, 'degradation_metadata.csv'))

            # saving essential pipeline hyperparameters
            pipeline_hyperparams = defaultdict(list)
            for key, operation in self.pipeline.items():
                op_hyperparams = operation.get_hyperparams()
                for hyperparam, val in op_hyperparams.items():
                    pipeline_hyperparams['index_num'].append(key[0])
                    pipeline_hyperparams['degradation'].append(key[1])
                    pipeline_hyperparams['hyperparam'].append(hyperparam)
                    pipeline_hyperparams['value'].append(val)

            if len(pipeline_hyperparams) > 0:
                df_hyperparams = pd.DataFrame.from_dict(pipeline_hyperparams).set_index(['index_num'])
                df_hyperparams.to_csv(os.path.join(save_to_dir, 'degradation_hyperparameters.csv'))

        # TODO: how to clear these files at end of epoch?
        # if self.jm_present:
        #     JMCompress.cleanup(self.jm_cleanup_files[-2:])  # files in the scratch directory are cleared separately

        # TODO: if given one image, return one image without a list, and return metadata without a dict...

        meta_keys = []  # TODO: this code needs to be arranged when random degradations are used
        ordered_keys = []

        for image, meta_dict in final_metadata.items():
            all_values = []

            if len(ordered_keys) == 0:
                ordered_keys = sorted(meta_dict.keys())

            for degradation in ordered_keys:
                value = meta_dict[degradation]
                if isinstance(value, list):
                    all_values.extend(value)
                    meta_keys.extend([degradation] * len(value))
                else:
                    all_values.append(value)
                    meta_keys.append(degradation)
            meta_vals = np.array(all_values)  # TODO: remove hardcoding...

        if len(final_images) == 1:
            final_images = final_images[0]

        return final_images, meta_vals, meta_keys


def pipeline_prep_and_run(pipeline_config, **kwargs):
    """
    Runs the image manipulation pipeline using the provided config file and command-line kwargs.
    """

    kwargs = {k: v for (k, v) in kwargs.items() if v is not None}  # filter out any None values

    if pipeline_config:  # extract configuration parameters from toml file
        config_params = convert_default_none_dict(toml.load(pipeline_config))

        if config_params['override_cli'] or kwargs['override_cli']:
            # config takes priority only if this is explicity requested
            config_params = {**kwargs, **config_params}
        else:
            # combine additional command-line and config file arguments (command-line takes priority)
            config_params = {**config_params, **kwargs}
    else:  # extract configuration parameters solely from command-line input
        config_params = kwargs
        config_params = convert_default_none_dict(config_params)

    source_dir = config_params['source_dir']
    output_dir = config_params['output_dir']
    recursive = config_params['recursive']

    if config_params['source_dir'] is None or config_params['output_dir'] is None:
        raise RuntimeError('Input/Output folders need to be defined at the command line or in the config file.')

    if config_params['pipeline'] is None:
        raise RuntimeError('Pipeline of operations needs to be defined in the config file or command arguments.')

    if isinstance(config_params['pipeline'], str):  # extract pipeline from direct command-line argument, if present
        config_params['pipeline'] = config_params['pipeline'].split('-')

    g_utils.set_random_seed(config_params['seed'])  # prepare random seeds

    create_dir_if_empty(output_dir)  # output setup

    if os.path.isdir(source_dir):
        image_names = extract_image_names_from_folder(source_dir, recursive=recursive)
    elif os.path.isfile(source_dir):
        image_names = [source_dir]
    else:
        raise RuntimeError('Please provide a valid filename/folder.')

    converter = ImagePipeline(**config_params)  # main pipeline
    converter.run_pipeline(image_files=image_names, save_to_dir=output_dir, multiples=config_params['multiples'])

    if converter.jm_present:
        clean_scratch_dir(os.path.dirname(converter.jm_cleanup_files[0]))  # removes any temp JM files from the scratch directory

    with open(os.path.join(output_dir, 'degradation_config.toml'), 'w') as f:
        toml.dump(config_params, f)


