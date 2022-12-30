from rumpy.SISR.models.SFTMD_variants.architectures import SFTMD
from rumpy.sr_tools.data_handler import read_augmentation_list
from SISR.models import BaseModel
import os
import torch
from torch import nn
import numpy as np


# TODO: needs a complete revamp!
class AttSplitHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, normalize=True, lr_data_loc=None, partitions=3,
                 lr_eval_data_loc=None, blur_kernels=False, blur_kernels_only=False, metadata=None,
                 specific_cut=False, data_attributes=None,
                 **kwargs):
        super(AttSplitHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                              **kwargs)
        if lr_data_loc is None:
            raise Exception('Location of compression parameters needs to be specified for this network to run.')

        if blur_kernels:
            kernel_loc = os.path.join(lr_data_loc, 'kernel_maps.npy')
        else:
            kernel_loc = None
        self.augmentation_list, *_ = read_augmentation_list(os.path.join(lr_data_loc, 'qpi_slices.csv'),
                                                            normalize=normalize, legacy_blur_kernels=kernel_loc,
                                                            no_qpi=blur_kernels_only, attributes_loc=data_attributes,
                                                            metadata=metadata)
        min = 0
        max = 1
        multiplier = 100
        cutoffs = list(range(min*multiplier, max*multiplier, int(((max*multiplier)-(min*multiplier))/partitions)+1)) + [multiplier+1]

        cutoffs = [c/multiplier for c in cutoffs]
        if specific_cut:  # TODO: remove this hardcode
            cutoffs = [0, 0.25, 0.5, 0.75, 1.1]
        self.net = MultiSFT(partitions=partitions, cutoffs=cutoffs, device=device, **kwargs)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.define_optimizer(lr=lr)
        self.model_name = 'attsplit'
        self.criterion = nn.L1Loss()

        if lr_eval_data_loc is not None:
            eval_list, _, _ = read_augmentation_list(os.path.join(lr_eval_data_loc, 'qpi_slices.csv'),
                                                     attributes_loc=data_attributes,
                                                     normalize=normalize, metadata=metadata)
            self.augmentation_list = {**self.augmentation_list, **eval_list}

    def collect_attributes(self, image_names):
        attributes = []
        for image in image_names:
            attributes.append(self.augmentation_list[image])
        return torch.tensor(attributes)

    def generate_channels(self, x, attributes):
        l_attributes = attributes.tolist()
        num_channels = np.array(l_attributes[0]).size
        extra_channels = torch.ones(x.size(0), num_channels, *x.size()[2:])

        for index, att in enumerate(l_attributes):
            if num_channels == 1:
                extra_channels[index, ...] = extra_channels[index, ...] * att
            else:
                extra_channels[index, ...] = torch.from_numpy(np.expand_dims(
                    np.expand_dims(att, axis=-1), axis=-1).repeat(x.size()[2], 1).repeat(
                    x.size()[3], 2))

        return extra_channels

    def run_train(self, x, y, image_names=None, masks=None, *args, **kwargs):
        attributes = self.collect_attributes(image_names).to(device=self.device)
        extra_channels = self.generate_channels(x, attributes).to(device=self.device)
        return super().run_train(x, y, attributes=attributes, image_names=image_names,
                                 masks=masks, extra_channels=extra_channels)

    def run_eval(self, x, y=None, request_loss=False, image_names=None, *args, **kwargs):
        attributes = self.collect_attributes(image_names).to(device=self.device)
        extra_channels = self.generate_channels(x, attributes).to(self.device)
        return super().run_eval(x, y, request_loss=request_loss, attributes=attributes, extra_channels=extra_channels)

    def run_model(self, x, extra_channels=None, attributes=None, *args, **kwargs):
        return self.net.forward(x, extra_channels=extra_channels, attributes=attributes)


class MultiSFT(nn.Module):
    def __init__(self, cutoffs, scale, device, partitions=3, **kwargs):
        super(MultiSFT, self).__init__()
        self.sub_nets = nn.ModuleList()
        self.cutoffs = cutoffs
        self.scale = scale
        self.device = device
        for i in range(partitions):
            self.sub_nets.append(SFTMD(scale=scale, **kwargs))

    def partition_input(self, x, extra_channels, attributes):
        x_slices = {}
        channel_slices = {}
        funnels = {}
        l_attributes = attributes.tolist()
        for index, (lower, upper) in enumerate(zip(self.cutoffs[:-1], self.cutoffs[1:])):
            funnel = [upper > att >= lower for att in l_attributes]
            if sum(funnel) > 0:
                x_slices[index] = x[funnel, ...]
                funnels[index] = funnel
                channel_slices[index] = extra_channels[funnel, ...]

        return x_slices, channel_slices, funnels

    def forward(self, x, extra_channels, attributes):
        x_slices, channel_slices, funnels = self.partition_input(x, extra_channels, attributes)
        result = torch.zeros(*x.size()[0:2], *[x * self.scale for x in x.size()[2:]]).to(device=x.device)

        for index in x_slices.keys():
            result[funnels[index], ...] = self.sub_nets[index].forward(x_slices[index], channel_slices[index])
        # TODO: add an option to cancel this change if network is in eval mode
        # TODO: This modification (where grad is set to None to prevent updates) makes it incompatible with multiple-GPU training.  Will need to move this out of the forward call.
        # TODO: However, this could actually warrant some further investigation, as not clear why GPU split has such a profound effect - only 1/2 batches will have non-matching QPIs in each GPU
        for i in range(len(self.sub_nets)):
            if i not in x_slices.keys():
                self.sub_nets[i].eval()
                for param in self.sub_nets[i].parameters():
                    param.grad = None

        return result


