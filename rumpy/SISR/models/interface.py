import glob
import os
import toml
import torch

from deepdiff import DeepDiff

from rumpy.shared_framework.models.base_interface import ImageModelInterface
from rumpy.sr_tools.helper_functions import read_metadata


class SISRInterface(ImageModelInterface):
    """
    Main SISR model interface.
    """
    def __init__(self, *args, scale=None, **kwargs):
        """
        :param scale: SR scale factor.
        For other general params, check ImageModelInterface.
        """
        self.scale = scale
        super(SISRInterface, self).__init__(*args, **kwargs)

    def _metadata_load(self, experiment, load_epoch, new_params, new_params_override_load, **kwargs):
        """
        Incorporates some additional checks for scale factor adherence.
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

        if hasattr(self, 'name') and self.name == 'qpircan':  # legacy conversion system
            self.name = 'qrcan'

        if self.metadata is not None and self.scale is not None and self.scale != self.metadata['internal_params']['scale']:
            raise Exception('The model loaded has been trained for a different scale, '
                            'and cannot produce the requested images.')

    @staticmethod
    def _legacy_model_setup(experiment, exp_folder, scale):
        """
        Specific legacy compatibility function for old SISR models. Should not be needed for newer models.
        :param experiment: Experiment name.
        :param exp_folder: Experiment folder.
        :param scale: SISR scale factor.
        :return: model metadata
        """
        metadata = {}
        try:
            l_data = read_metadata(os.path.join(exp_folder, 'meta_data.csv'))
        except Exception:
            raise RuntimeError('No metadata information provided - model structure unknown.')
        metadata['name'] = l_data['model']
        metadata['internal_params'] = {}
        metadata['internal_params']['scale'] = scale
        if experiment == 'SFTMD_256_T1' or 'EDSR_MD_T1':
            num_feats = 256
            metadata['internal_params']['num_feats'] = num_feats
        if experiment == 'EDSR_MD_T1':
            metadata['internal_params']['normalize'] = False
        if experiment == 'EDSR_T1_x8':
            metadata['internal_params']['scale'] = 8
            metadata['internal_params']['num_features'] = 256
            metadata['internal_params']['num_blocks'] = 32
        return metadata

    def train_batch(self, lr, hr, *args, **kwargs):
        """
        main model training interface - input is an LR image, GT is an HR image.
        """
        return self.model.run_train(x=lr, y=hr, **kwargs)

    def net_run_and_process(self, lr=None, hr=None, **kwargs):
        """
        Main eval function - will also properly post-process images for downstream usage.
        :return: Output RGB and YCbCr image, along with loss and timing values if requested.
        """
        # TODO: allow user to prevent preprocessing from happening?
        # TODO: add checks or fix dependency on input type
        if 'rgb' in self.configuration['colorspace']:
            out_rgb, loss, timing = self.model.run_eval(x=lr, y=hr, **kwargs)
            out_ycbcr = self.colorspace_convert(out_rgb, colorspace='rgb')
            out_rgb = self._standard_image_formatting(out_rgb.numpy())
        else:
            if hr is None:  # TODO: better fix possible?
                f_ref = hr
            else:
                f_ref = hr[:, 0, :, :].unsqueeze(1)
            out_y, loss, timing = self.model.run_eval(lr[:, 0, :, :].unsqueeze(1), y=f_ref, **kwargs)
            out_ycbcr = torch.stack([out_y.squeeze(1), lr[:, 1, :, :], lr[:, 2, :, :]], 1)
            out_rgb = self.colorspace_convert(out_ycbcr, colorspace='ycbcr')
            out_ycbcr = self._standard_image_formatting(out_ycbcr.numpy())

        return out_rgb, out_ycbcr, loss, timing

    def net_forensic(self, data, **kwargs):
        """
        Main forensic network call - in this case produces the final image output and any additional diagnostic info.
        """
        image, forensic_data = self.model.run_forensic(data, **kwargs)
        return image.numpy(), forensic_data
