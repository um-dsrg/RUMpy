import os

import numpy as np
import torch
from rumpy.shared_framework.models.base_architecture import BaseModel

from rumpy.regression.models.contrastive_learning.encoding_models import Encoder
from rumpy.regression.models.contrastive_learning.moco import MoCo


class QModel(BaseModel):
    """
    Model template for use when modulating typical SR networks with metadata.
    """
    def __init__(self, metadata=None, use_moco=None, pre_trained_encoder_weights=None, metadata_bypass_len=None,
                 ignore_degradation_location=False, **kwargs):
        self.style = None  # only relevant to QRCAN
        self.channel_concat = False  # only relevant to models concatenating extra channels with input
        self.no_metadata = False
        self.metadata_keys_used_in_training = None
        self.ignore_degradation_location = ignore_degradation_location

        if metadata_bypass_len:
            self.num_metadata = metadata_bypass_len
            self.metadata = None
        else:
            if metadata is not None:
                self.num_metadata = len(metadata)

                if 'contrastive_encoding' in metadata:
                    self.num_metadata += 255
                if 'contrastive_q' in metadata:
                    self.num_metadata += 255
                if 'contrastive_encoding_tsne' in metadata:
                    self.num_metadata += 1
                if 'contrastive_q_tsne' in metadata:
                    self.num_metadata += 1
                if 'contrastive_encoding_pca' in metadata:
                    self.num_metadata += 10
                if 'contrastive_q_pca' in metadata:
                    self.num_metadata += 7


                if 'all' in metadata:
                    self.num_metadata += 39  # all celeba attributes

                if 'blur_kernel' in metadata:
                    self.num_metadata += 9
                elif 'unmodified_blur_kernel' in metadata or any(['unmodified_blur_kernel' in meta_op for meta_op in metadata]):
                    self.num_metadata += 440

                self.metadata = metadata
                if self.ignore_degradation_location:  # removes encoding of degradation location
                    self.metadata = [m[2:] if m[0].isdigit() else m for m in self.metadata]
            else:
                if self.no_metadata:
                    self.metadata = None
                    self.num_metadata = 0
                else:
                    self.metadata = ['qpi']
                    self.num_metadata = 1

        super(QModel, self).__init__(**kwargs)

        if use_moco:  # includes a contrastive learning encoder
            self.moco_encoder = MoCo(base_encoder=Encoder)

            for param in self.moco_encoder.parameters():  # currently, MoCo will be frozen during training
                param.requires_grad = False
            self.moco_encoder.eval()
            self.moco_encoder.to(kwargs['device'])
            if kwargs['device'] == torch.device('cpu'):  # device location to map to, when pre-loading
                loc = kwargs['device']
            else:
                loc = "cuda:%d" % kwargs['device']

            # state = torch.load(f=pre_trained_encoder_weights, map_location=loc)
            # self.moco_encoder.load_state_dict(state_dict=state['network'])
            self.num_metadata = 256  # fixed value
            self.moco_encoding = True
            self.no_metadata = True
            self.channel_concat = False
        else:
            self.moco_encoding = False

    def generate_channels(self, x, metadata, keys):
        """
        Specific function used to morph metadata into format required for Q-layer blocks.
        """
        if metadata is None:
            raise RuntimeError('Metadata needs to be specified for this network to run properly.')
        extra_channels = torch.ones(x.size(0), self.num_metadata)
        if 'all' in self.metadata:
            mask = [True] * self.num_metadata
        else:
            mask = [True if key[0] in self.metadata else False for key in keys]

        for index, _ in enumerate(extra_channels):
            if len(keys) == 1:  # TODO: any way to shorten this?
                added_info = metadata[index]
            else:
                added_info = metadata[index][mask]
            extra_channels[index, ...] = extra_channels[index, :] * added_info
        extra_channels = extra_channels.unsqueeze(2).unsqueeze(3)
        if self.style == 'modulate':
            extra_channels = self.scale_qpi(extra_channels)
        return extra_channels

    def generate_sft_channels(self, x, metadata, metadata_keys):
        """
        Specific function used to morph metadata into format required for SFT blocks.
        """
        if metadata is None:
            raise RuntimeError('Metadata needs to be specified for this network to run properly.')
        extra_channels = torch.ones(x.size(0), self.num_metadata, *x.size()[2:])
        if extra_channels.device != metadata.device:
            extra_channels = extra_channels.to(metadata.device)
        mask = [True if key[0] in self.metadata else False for key in metadata_keys]
        for index, _ in enumerate(extra_channels):

            if len(metadata_keys) == 1:  # TODO: any way to shorten this?
                added_info = metadata[index]
            else:
                added_info = metadata[index][mask]
            if self.num_metadata == 1:
                extra_channels[index, ...] = extra_channels[index, ...] * added_info
            else:
                if isinstance(metadata, np.ndarray):
                    extra_channels[index, ...] = torch.from_numpy(np.expand_dims(
                        np.expand_dims(added_info, axis=-1), axis=-1).repeat(x.size()[2], 1).repeat(
                        x.size()[3], 2))
                else:
                    extra_channels[index, ...] = torch.unsqueeze(torch.unsqueeze(added_info, -1), -1).repeat_interleave(
                        x.size()[2], 1).repeat_interleave(x.size()[3], 2)

        return extra_channels

    def channel_concat_logic(self, x, extra_channels, metadata, metadata_keys):
        """
        Main channel concatenation stage.
        Metadata needs to be selectively filtered and converted to the correct format for the model to use.
        :param x: Input image batch (N, C, H, W)
        :param extra_channels: Optional pre-prepared metadata channels.  Can be set to None to ignore.
        :param metadata: Metadata information, for each image in batch provided (N, M).
        :param metadata_keys: List of keys corresponding to metadata, to allow for selective filtering.
        :return: Modulated input batch (if required), modulated metadata ready for model use.
        """
        if self.no_metadata:
            extra_channels = None
        else:
            if extra_channels is None:
                extra_channels = self.generate_channels(x, metadata, metadata_keys)

                if not self.channel_concat and self.device != extra_channels.device:
                    extra_channels = extra_channels.to(self.device)

            if self.metadata_keys_used_in_training is None:
                self.metadata_keys_used_in_training = [m[0] for m in metadata_keys]
        if self.channel_concat:
            input_data = torch.cat((x, extra_channels), 1)
        else:
            input_data = x

        return input_data, extra_channels

    def save_model(self, model_save_name, extract_state_only=True, minimal=False):
        super().save_model(model_save_name=model_save_name,
                           extract_state_only=extract_state_only,
                           minimal=minimal)

        if self.metadata_keys_used_in_training:
            self.state['metadata_keys_used_in_training'] = self.metadata_keys_used_in_training

        torch.save(self.state, f=os.path.join(self.model_save_dir, "{}_{}".format(model_save_name, self.curr_epoch)))

    def load_model(self, model_save_name, model_idx, legacy=False, load_override=None, preloaded_state=None):
        state = super().load_model(model_save_name=model_save_name,
                                   model_idx=model_idx,
                                   legacy=legacy,
                                   load_override=load_override,
                                   preloaded_state=preloaded_state)

        return state

    def run_train(self, x, y, metadata=None, extra_channels=None, metadata_keys=None, *args, **kwargs):
        input_data, extra_channels = self.channel_concat_logic(x, extra_channels, metadata, metadata_keys)
        return super().run_train(input_data, y, extra_channels=extra_channels, **kwargs)

    def run_eval(self, x, y=None, request_loss=False, metadata=None, metadata_keys=None,
                 extra_channels=None, *args, **kwargs):
        input_data, extra_channels = self.channel_concat_logic(x, extra_channels, metadata, metadata_keys)
        return super().run_eval(input_data, y, request_loss=request_loss, extra_channels=extra_channels, **kwargs)

    def run_forensic(self, x, metadata=None, metadata_keys=None, extra_channels=None, *args, **kwargs):
        input_data, extra_channels = self.channel_concat_logic(x, extra_channels, metadata, metadata_keys)
        # return super().run_forensic(input_data, qpi=extra_channels)
        return super().run_forensic(input_data, metadata=extra_channels)

    def run_model(self, x, extra_channels=None, *args, **kwargs):
        if self.moco_encoding:
            extra_channels = self.moco_encoder.forward(x, x).unsqueeze(2).unsqueeze(3)
        return self.net.forward(x, metadata=extra_channels)
