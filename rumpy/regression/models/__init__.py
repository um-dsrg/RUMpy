import torch
from torch import nn as nn
from torchvision.transforms import CenterCrop
import pandas as pd
import json


from rumpy.image_tools.image_manipulation.image_functions import image_patch_selection
from rumpy.shared_framework.models.base_architecture import BaseModel
from rumpy.sr_tools.loss_functions import OccupancyLoss


class SelectiveSoftmax(nn.Module):
    def __init__(self, softmax_range):
        super(SelectiveSoftmax, self).__init__()
        self.softmax_range = softmax_range
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        soft_out = x.clone()
        sel_soft = self.softmax(x[:, self.softmax_range[0]:self.softmax_range[1]])
        soft_out[:, self.softmax_range[0]:self.softmax_range[1]] = sel_soft
        return soft_out


class DegradationRegressor(BaseModel):
    """
    Basic Regressor Model Parameters.
    """
    def __init__(self, device, model_save_dir, eval_mode=False, input_patch_num=1, centercrop_patch_eval=True,
                 crop_size=200, normalization_scheme=None, normalization_params=None, occupancy_loss=False,
                 patch_selection_strategy='random', occ_weight=1.0, l1_weight=1.0, predefined_patches=None, **kwargs):
        super(DegradationRegressor, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                                   **kwargs)

        self.input_patch_num = input_patch_num
        self.crop_size = crop_size
        self.centercrop_patch_eval = centercrop_patch_eval
        self.normalization_scheme = normalization_scheme
        self.patch_selection_strategy = patch_selection_strategy
        self.eval_request_loss = True  # loss can be calculated on eval data

        if predefined_patches and not self.eval_mode:  # TODO: eventually this should be cut out from here and taken care of solely by data handler
            self.predefined_patches = {}
            patches = pd.read_csv(predefined_patches, header=0, index_col=0).to_dict()['high_entropy_patches_left_corner']
            for key, val in patches.items():
                self.predefined_patches[eval(key)[0]] = eval(val)
        else:
            self.predefined_patches = None

        self.norm_params = {}

        if normalization_scheme:
            if not normalization_params:
                raise RuntimeError('Normalization parameters (mean, max etc.) need to be specified if normalization is required.')

            for key, data in normalization_params.items():
                self.norm_params[key] = torch.tensor(data).to(device=device)

        if occupancy_loss:
            if normalization_scheme:
                self.occ_loss = OccupancyLoss(device=device,
                                              zero_thres=(1e-6 - normalization_params['mean'])/normalization_params['std'])
            else:
                self.occ_loss = OccupancyLoss(device=device)

            self.l1_weight = l1_weight
            self.occ_weight = occ_weight
        else:
            self.occ_loss = None

        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.regressor_type = 'standard'

    def norm(self, x):
        if self.normalization_scheme == 'zero_mean':
            y = (x - self.norm_params['mean']) / self.norm_params['std']
        elif self.normalization_scheme == 'zero_to_one':
            y = (x - self.norm_params['minim']) / (self.norm_params['maxim'] - self.norm_params['minim'])
        else:
            y = x
        return y

    def run_train(self, x, y, *args, **kwargs):

        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')

        # catering for training norm (model should only see normalized ground-truth labels)
        x, y = x.to(device=self.device), y.to(device=self.device)
        y = self.norm(y)

        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)

        out = self.run_model(x, **kwargs)  # run data through model

        if self.occ_loss:
            loss_l1 = self.criterion(out, y)
            loss_occ = self.occ_loss(out, y)
            loss = self.occ_weight * loss_occ + self.l1_weight * loss_l1

            loss_package = {}

            for _loss, name in zip((loss, loss_l1, loss_occ),
                                   ('train-loss', 'l1-loss', 'occ-loss')):
                loss_package[name] = _loss.cpu().data.numpy()

        else:
            loss = self.criterion(out, y)
            loss_package = loss.detach().cpu().numpy()

        self.standard_update(loss)  # takes care of optimizer calls, backprop and scheduler calls

        return loss_package, out.detach().cpu()

    def run_eval(self, x, y=None, keep_on_device=False, *args, **kwargs):

        if self.centercrop_patch_eval:
            if x.shape[2] > self.crop_size and x.shape[3] > self.crop_size:  # crops out a central section from the input image
                x = CenterCrop(size=self.crop_size)(x)

        elif self.input_patch_num > 1:  # extracts and combines multiple random patches from input image (this is taken care of within the data handler for the training phase)
            # TODO: this system is incapable of dealing with a batch size higher than 1

            if not self.eval_mode and self.predefined_patches:
                patch_strategy = 'predefined'
                patch_locations = self.predefined_patches[kwargs['tag'][0]][0:self.input_patch_num]
            else:
                patch_strategy = self.patch_selection_strategy
                patch_locations = None

            lr_patches, _, _ = image_patch_selection(x.squeeze(), self.crop_size, scale=1,
                                                     patch_type=patch_strategy, predefined_patch_locations=patch_locations,
                                                     number_of_patches=self.input_patch_num)
            lr_patches = torch.stack(lr_patches, 0)
            stacked_patches = torch.zeros(lr_patches.size(0) * lr_patches.size(1), lr_patches.size(2), lr_patches.size(3))
            for i in range(self.input_patch_num):
                stacked_patches[i*3:(i*3)+3, ...] = lr_patches[i, ...]  # TODO: reconfirm this
            x = stacked_patches.unsqueeze(0)

        if y is not None:  # catering for eval norm (model should only see normalized ground-truth labels, but final output should be again un-normalized)
            y = y.to(device=self.device)
            y = self.norm(y)

        out, loss, timing = super().run_eval(x, y=y, keep_on_device=True, **kwargs)

        if self.normalization_scheme == 'zero_mean':
            out = (out * self.norm_params['std']) + self.norm_params['mean']
        elif self.normalization_scheme == 'zero_to_one':
            out = (out*(self.norm_params['maxim'] - self.norm_params['minim'])) + self.norm_params['minim']

        return out.cpu(), loss, timing
