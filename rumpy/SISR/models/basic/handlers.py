from rumpy.SISR.models.basic.architectures import SRCNN, VDSR
from rumpy.shared_framework.models.base_architecture import BaseModel
from torch import nn


class SRCNNHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, kernel_pattern=None, channel_pattern=None,
                 padding='same', scheduler=None, scheduler_params=None, perceptual=None, **kwargs):
        super(SRCNNHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                           **kwargs)
        self.net = SRCNN(kernel_pattern=kernel_pattern, channel_pattern=channel_pattern, padding=padding)
        self.colorspace = 'ycbcr'
        self.im_input = 'interp'
        self.criterion = nn.MSELoss()
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)
        self.model_name = 'srcnn'


class VDSRHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, kernel_pattern=None, channel_pattern=None,
                 padding='same', grad_clip=0.1, scheduler=None, scheduler_params=None, perceptual=None, **kwargs):
        super(VDSRHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                          grad_clip=grad_clip, **kwargs)
        if kernel_pattern is None:  # default structure for VDSR
            kernel_pattern = [3] * 20
        if channel_pattern is None:
            channel_pattern = [1] + [64] * 19 + [1]
        self.net = VDSR(kernel_pattern=kernel_pattern, channel_pattern=channel_pattern, padding=padding)
        self.colorspace = 'ycbcr'
        self.im_input = 'interp'
        self.criterion = nn.MSELoss()
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)
        self.model_name = 'vdsr'
