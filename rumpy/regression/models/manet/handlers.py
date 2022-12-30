import torch
import torchvision.models as models
import torch.nn as nn

from rumpy.regression.models import SelectiveSoftmax, DegradationRegressor
from rumpy.regression.models.manet.architectures import MANet


class ManetHandler(DegradationRegressor):
    """
    MANet - a specific model for blur kernel prediction (from https://arxiv.org/pdf/2108.05302.pdf)
    ** Still under construction **
    """

    def __init__(self, device, model_save_dir, eval_mode=False, kernel_size=21, sr_scale=4,
                 scheduler=None, scheduler_params=None, invariant_kernel=False, lr=1e-4, **kwargs):

        super(ManetHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                           **kwargs)

        self.net = MANet(kernel_size=kernel_size, scale=sr_scale)

        self.invariant_kernel = invariant_kernel
        self.sr_scale = sr_scale

        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, device=device, perceptual=None)

    def spread_invariant_kernel(self, kernel, new_im_shape):
        return torch.tile(kernel.unsqueeze(2).unsqueeze(3), (1, 1, new_im_shape[0], new_im_shape[1]))

    def run_train(self, x, y, *args, **kwargs):
        if self.invariant_kernel:
            y = self.spread_invariant_kernel(y, (x.size(2)*self.sr_scale, x.size(3)*self.sr_scale))
        return super().run_train(x, y, *args, **kwargs)

    def run_eval(self, x, y=None, *args, **kwargs):
        if self.invariant_kernel and y is not None:
            y = self.spread_invariant_kernel(y, (x.size(2)*self.sr_scale, x.size(3)*self.sr_scale))
        return super().run_eval(x, y, *args, **kwargs)
