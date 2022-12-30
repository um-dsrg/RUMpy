import os

import numpy as np
import pandas as pd
import toml
import torch
from colorama import init, Fore
from torchinfo import summary

from collections import defaultdict

from deepdiff import DeepDiff

from rumpy.image_tools.image_manipulation.image_functions import ycbcr_convert
from rumpy.shared_framework.models import define_model
from rumpy.sr_tools.helper_functions import create_dir_if_empty
from rumpy.sr_tools.stats import load_statistics
from rumpy.shared_framework.configuration.gpu_check import device_selector

init()  # colorama setup


class ImageModelInterface:
    """
    Main model client-side interface.  Takes care of loading/saving models,
    formatting outputs and triggering training/eval.

    Application-specific interfaces should inherit from this base class.
    """
    def __init__(self, model_loc, experiment, gpu='off', sp_gpu=0, mode='eval', new_params=None,
                 load_epoch=None, save_subdir=None, best_load_metric='val-PSNR', no_directories=False,
                 new_params_override_load=None, loss_masking=False, skip_scheduler_load=False, skip_optimizer_load=False):
        """
        :param model_loc: Model directory location.
        :param experiment: Model experiment name.
        :param gpu: One of 'off', 'multi' or 'single'.  Off signals models to use the CPU, single allows
        use of one GPU, and multi will trigger the use of all available GPUs.
        :param sp_gpu: Select specific GPU to use, if more than one are available.
        :param mode: Either eval or train.
        :param new_params: New model parameter dictionary, if creating a new model.
        :param load_epoch:  Epoch number to load, if reading model weights from file.
        :param save_subdir: Save subdirectory if branching is in use.
        :param best_load_metric: Sets metric to use when loading models using 'best'
        :param no_directories: Set to true to prevent model from creating a disk directory
        :param new_params_override_load: Set to true to force model config of new loaded model to match that
         of the new params specified.
        :param skip_scheduler_load: Set to true to skip loading the scheduler when loading a saved model.
        :param skip_optimizer_load: Set to true to skip loading the optimizer when loading a saved model.
        :param loss_masking: Set to true to force model to accept HR mask for loss function masking
        """

        log_dir = 'result_outputs'
        save_dir = 'saved_models'
        self.mode = mode

        # GPU setup
        self.device = device_selector(gpu, sp_gpu)

        # inner experiment file path setup
        self.experiment = experiment

        self.base_folder, self.logs, self.saved_models = self.prepare_standard_paths(log_dir, save_dir, experiment,
                                                                                     model_loc, save_subdir)

        # specific mode checks
        if mode == 'train':
            if not no_directories:
                create_dir_if_empty(self.base_folder, self.logs, self.saved_models)
            if new_params is None and load_epoch is None:
                raise RuntimeError('Need to specify model parameters to train a new model.')
        elif mode == 'eval':
            if load_epoch is None:
                raise RuntimeError('Need to specify which model epoch to load.')

        # Dictionary which tracks any changes between loaded config files (for continuing training)
        self.config_changes = None

        self._metadata_load(experiment, load_epoch, new_params, new_params_override_load)  # model config loading

        self.model = define_model(name=self.name, model_save_dir=self.saved_models,
                                  device=self.device, eval_mode=True if mode == 'eval' else False,
                                  checkpoint_load=True if load_epoch is not None else False,
                                  loss_masking=loss_masking,
                                  **self.metadata['internal_params'])  # main model definition function

        if load_epoch is not None:
            stats_path = os.path.join(self.logs, 'summary.csv')
            if os.path.isfile(stats_path):
                self.stats = load_statistics(self.logs, 'summary.csv', config='pd')  # loads model training stats
                if load_epoch == 'best':
                    load_epoch = self.model.best_model_selection_criteria(stats=self.stats,
                                                                          base_metric=best_load_metric,
                                                                          model_metadata=self.metadata['internal_params'])
                elif load_epoch == 'last':
                    load_epoch = len(self.stats[best_load_metric]) - 1
            else:
                print('%sNo training stats found for %s %s' % (Fore.RED, self.experiment, Fore.RESET))

            self.model_epoch = load_epoch

            # TODO: add system which remembers which networks were trained in the old style, to turn on/off legacy var
            if self.config_changes is not None:
                self.model.load_model(model_save_name='train_model', model_idx=load_epoch, legacy=self.model.legacy_load, config_changes=self.config_changes, skip_scheduler_load=skip_scheduler_load, skip_optimizer_load=skip_optimizer_load)
            else:
                if mode == 'train':
                    self.model.load_model(model_save_name='train_model', model_idx=load_epoch, legacy=self.model.legacy_load, skip_scheduler_load=skip_scheduler_load, skip_optimizer_load=skip_optimizer_load)
                else:
                    self.model.load_model(model_save_name='train_model', model_idx=load_epoch, legacy=self.model.legacy_load)
        else:
            self.model.pre_training_model_load()

        self.full_name = '%s_%d' % (experiment, self.model_epoch)

        if gpu == 'multi':
            self.model.set_multi_gpu()

        im_input = self.model.im_input
        colorspace = self.model.colorspace

        # No information whether model is SISR or multi-frame; default to SISR (legacy compatibility)
        if not hasattr(self.model, 'model_type'):
            self.model.model_type = 'single-frame'

        self.configuration = {'input': im_input, 'colorspace': colorspace, 'model_type': self.model.model_type}

        self.print_overview()

    @staticmethod
    def prepare_standard_paths(log_dir, save_dir, experiment, model_loc, save_subdir):
        """
        Prepares standard file path locations for specified model.
        :param log_dir: Log directory name (typically 'result_outputs').
        :param save_dir: Actual model weight directory (typically 'saved_models').
        :param experiment: Model experiment name.
        :param model_loc: Base model location.
        :param save_subdir: Branch base name, if in use.
        :return: Base experiment folder, logs folder and saved models folder.
        """
        if save_subdir:
            base_folder = os.path.abspath(os.path.join(model_loc, experiment, save_subdir))
        else:
            base_folder = os.path.abspath(os.path.join(model_loc, experiment))
        logs = os.path.abspath(os.path.join(base_folder, log_dir))
        saved_models = os.path.abspath(os.path.join(base_folder, save_dir))

        return base_folder, logs, saved_models

    def init_new_branch(self, branch_name):
        """
        Creates a new sub-branch by updating folder output locations.
        :param branch_name: Name of new branch.
        :return: None
        """
        self.base_folder = os.path.join(self.base_folder, branch_name)
        self.logs = os.path.join(self.base_folder, 'result_outputs')
        self.saved_models = os.path.join(self.base_folder, 'saved_models')
        self.model.model_save_dir = self.saved_models
        create_dir_if_empty(self.base_folder, self.logs, self.saved_models)

    def defaultdict_to_standard_dict(self, dictionary):
        """
        Recursively convert all defcultdicts to dicts.

        Source: https://stackoverflow.com/a/26496899
        """
        if isinstance(dictionary, defaultdict):
            dictionary = {key: self.defaultdict_to_standard_dict(value) for key, value in dictionary.items()}
        return dictionary

    def _metadata_load(self, experiment, load_epoch, new_params, new_params_override_load, **kwargs):
        """
        Loads metadata from file, or from provided new parameters.
        """
        if load_epoch is None:
            self.model_epoch = 0
            self.metadata = new_params
        else:
            if os.path.exists(os.path.join(self.base_folder, 'config.toml')):
                original_params = toml.load(os.path.join(self.base_folder, 'config.toml'))['model']
                new_params_converted = self.defaultdict_to_standard_dict(new_params)

                param_diff = DeepDiff(original_params, new_params_converted, ignore_type_in_groups=[(int, float)])

                # For now, check the values that have been changed
                # TODO: Do we need to check the values that have been added/removed?
                if 'values_changed' not in param_diff:
                    if new_params_override_load:
                        self.metadata = new_params
                    else:
                        self.metadata = original_params
                else:
                    if new_params_override_load is None:
                        raise RuntimeError('There are parameter inconsistencies between the current config and the saved-model config in %s. ' % os.path.join(self.base_folder, 'config.toml') +\
                                           'Please set the argument new_params_override_load under the [training] section to True ' +\
                                           'to use the parameters of the current config, or to False to use the parameters from the original config.\n' +\
                                           'Difference between parameter dictionaries: %s.' % str(param_diff))
                    elif new_params_override_load:
                        self.metadata = new_params
                        self.config_changes = param_diff
                    else:
                        self.metadata = original_params
            else:
                self.metadata = new_params

        if self.metadata is not None and 'name' in self.metadata:
            self.name = self.metadata['name'].lower()

    @staticmethod
    def colorspace_convert(image, colorspace='rgb'):
        processed_im = ImageModelInterface._standard_image_formatting(image.numpy())
        for i in range(processed_im.shape[0]):
            processed_im[i, ...] = ycbcr_convert(processed_im[i, ...], im_type='jpg', input=colorspace, y_only=False)
        return processed_im

    @staticmethod
    def _standard_image_formatting(im, min_value=0, max_value=1):
        """
        Standardizes an image by clipping to specified min/max values
        """
        im_batch = np.copy(im)
        im_batch = np.clip(im_batch, min_value, max_value)
        return im_batch

    def train_batch(self, *args, **kwargs):
        """
        main model training interface
        """
        raise NotImplementedError('Training routine needs to be implemented for each individual application.')

    def net_run_and_process(self, **kwargs):
        raise NotImplementedError('Eval routine needs to be implemented for each application.')

    def net_forensic(self, data, **kwargs):
        raise NotImplementedError('Forensic routine needs to be implemented for each application.')

    def save(self, name='train_model', override=False, dry_run=False, minimal=False):
        """
        Saves current model to file.
        :param name: Output model name.
        :param override: Set to true to overwrite files when saving.
        :param dry_run: Set to true to only test if saving is possible, without actually saving to file.
        :param minimal: Set to true to only save eval portion of model, discarding optimizers etc.
        """

        if not minimal:
            name_prepend = name
        else:
            name_prepend = name + "_minimal"

        full_name = "{}_{}".format(name_prepend, str(self.model_epoch))

        save_path = os.path.join(self.saved_models, full_name)

        if os.path.isfile(save_path) and not override:
            raise RuntimeError('Saving this model will result in overwriting existing data!  '
                               'Change model location or enable override.')

        if not dry_run:
            self.model.save_model(model_save_name=name_prepend, minimal=minimal)
        else:
            print('Training cleared to run.')

    def save_metadata(self):
        """
        Function that saves any pertinent metadata that isn't obvious from the config file.
        """
        metadata = {'model_parameters': [self.model.print_parameters()]}
        md = pd.DataFrame.from_dict(metadata)
        md.to_csv(os.path.join(self.base_folder, 'extra_metadata.csv'), index=False)

        # saves entire model architecture summary using torchinfo
        # (does not actually call model.train or model.eval as no input data is being provided)
        model_structure = summary(self.model, mode='train', depth=5, device=self.device, verbose=0)
        with open(os.path.join(self.base_folder, 'model_structure.txt'), 'w', encoding='utf-8') as f:
            f.write(str(model_structure))

    def print_overview(self):
        """
        Function that prints out model diagnostic information.
        :return: None.
        """
        if self.mode == 'eval':
            pmode = 'eval'
            epoch = self.model_epoch
            message = 'currently evaluating'
        else:
            pmode = 'train'
            if self.model_epoch == 0:
                epoch = self.model_epoch
            else:
                epoch = self.model_epoch + 1
            message = 'will start training from'

        print('----------------------------')
        print('Handler for experiment %s initialized successfully.' % self.experiment)
        print('System loaded in %s mode - %s architecture provided.' % (pmode, self.name))
        print('Model has %d trainable parameters.' % self.model.print_parameters())
        if str(self.model.device) == 'cpu':
            device = self.model.device
        else:
            device = 'GPU ' + str(self.model.device)
        print("Using %s as the model's primary device, and %s "
              "epoch %d of the model." % (device, message, epoch))
        self.model.extra_diagnostics()
        print('----------------------------')

    def epoch_end_calls(self):
        self.model.epoch_end_calls()

    def get_learning_rate(self):
        return self.model.get_learning_rate()

    def set_epoch(self, epoch):
        self.model_epoch = epoch
        self.model.set_epoch(epoch)
