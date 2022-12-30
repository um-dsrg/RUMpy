from .architectures import EDSR, RCAN, SAN, HAN, SRMD, ELAN
from rumpy.shared_framework.models.base_architecture import BaseModel
from rumpy.SISR.models.attention_manipulators import QModel
import time
import torch


class EDSRHandler(BaseModel):
    """
    Main handler for EDSR.  Model-specific inputs found within architecture class.
    """

    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, in_features=3, hr_data_loc=None,
                 scheduler=None, scheduler_params=None, perceptual=None,
                 num_features=64, num_blocks=16, res_scale=0.1, **kwargs):
        super(EDSRHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                          hr_data_loc=hr_data_loc, **kwargs)
        self.net = EDSR(scale=scale, in_features=in_features, net_features=num_features, num_blocks=num_blocks,
                        res_scale=res_scale)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)

        self.model_name = 'edsr'


class RCANHandler(BaseModel):
    """
    Main handler for RCAN.  Most parameters are locked to ensure compliance with original model.
    """

    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, in_features=3, perceptual=None,
                 scheduler=None, scheduler_params=None, **kwargs):
        super(RCANHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                          **kwargs)
        self.net = RCAN(scale=scale, in_feats=in_features, **kwargs)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)
        self.model_name = 'rcan'


class HANHandler(BaseModel):
    """
    Main handler for HAN.  Most parameters are locked to ensure compliance with original model.
    """

    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, perceptual=None,
                 scheduler=None, scheduler_params=None, **kwargs):
        super(HANHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                         **kwargs)
        self.net = HAN(scale=scale)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)
        self.model_name = 'han'


class SANHandler(BaseModel):
    """
    Main handler for SAN.  Most parameters are locked to ensure compliance with original model.
    Since this model requires a lot of memory, the max_combined_im_size allows you to set the maximum
    patch size (width * height) allowed within the model.  If this is exceeded, the model will chop up the image
    into smaller patches, super-resolve them separately, and then stitch them together again.
    """

    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, perceptual=None,
                 max_combined_im_size=160000, scheduler=None, scheduler_params=None, **kwargs):
        super(SANHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                         **kwargs)
        self.net = SAN(scale=scale)
        self.scale = scale
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)

        self.max_combined_im_size = max_combined_im_size

        self.model_name = 'san'

    def forward_chop(self, x, shave=10):
        # modified from https://github.com/daitao/SAN/blob/master/TestCode/code/model/__init__.py

        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave

        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < self.max_combined_im_size:
            sr_list = []
            for chunk in lr_list:
                sr_list.append(super().run_eval(chunk, request_loss=False)[0])
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave)
                for patch in lr_list]

        h, w = self.scale * h, self.scale * w
        h_half, w_half = self.scale * h_half, self.scale * w_half
        h_size, w_size = self.scale * h_size, self.scale * w_size
        shave *= self.scale

        output = x.new(b, c, h, w)

        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def run_eval(self, x, y=None, request_loss=False, metadata=None, metadata_keys=None, timing=False, *args, **kwargs):
        if timing:
            tic = time.perf_counter()
        sr_image = self.forward_chop(x)
        if timing:
            toc = time.perf_counter()
        if request_loss:
            return sr_image, self.criterion(sr_image, y), toc - tic if timing else None
        else:
            return sr_image, None, toc - tic if timing else None


class EDSRMDHandler(QModel):
    """
    Implementation of EDSR which also allows one to insert metadata akin to SRMD.
    """

    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scheduler=None, scheduler_params=None,
                 num_features=256, num_blocks=32, scale=4, in_features=3, perceptual=None, **kwargs):
        super(EDSRMDHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                            **kwargs)
        self.net = EDSR(scale=scale, in_features=in_features + self.num_metadata, net_features=num_features,
                        num_blocks=num_blocks)
        self.colorspace = 'augmented_rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)
        self.model_name = 'edsrmd'
        self.channel_concat = True  # signal to concatenate channels for input

    def run_train(self, x, y, metadata=None, metadata_keys=None, *args, **kwargs):
        extra_channels = self.generate_sft_channels(x, metadata, metadata_keys)
        return super().run_train(x, y, extra_channels=extra_channels, **kwargs)

    def run_eval(self, x, y=None, metadata=None, metadata_keys=None, request_loss=False, *args, **kwargs):
        extra_channels = self.generate_sft_channels(x, metadata, metadata_keys)
        return super().run_eval(x, y, extra_channels=extra_channels, request_loss=request_loss, **kwargs)

    def run_model(self, x, *args, **kwargs):
        return self.net.forward(x)


class SRMDHandler(QModel):
    """
    Main implementation of SRMD.  Most features are locked to ensure compliance with original model.
    """

    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scheduler=None, scheduler_params=None,
                 in_features=3, perceptual=None, **kwargs):
        super(SRMDHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                          **kwargs)
        self.net = SRMD(in_nc=in_features + self.num_metadata, **kwargs)
        self.colorspace = 'augmented_rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)
        self.model_name = 'srmd'
        self.channel_concat = True  # signal to concatenate channels for input
        self.legacy_load = False  # legacy loading does not work with SRMD

    def run_train(self, x, y, metadata=None, metadata_keys=None, *args, **kwargs):
        extra_channels = self.generate_sft_channels(x, metadata, metadata_keys)
        return super().run_train(x, y, extra_channels=extra_channels, **kwargs)

    def run_eval(self, x, y=None, metadata=None, metadata_keys=None, request_loss=False, *args, **kwargs):
        extra_channels = self.generate_sft_channels(x, metadata, metadata_keys)
        return super().run_eval(x, y, extra_channels=extra_channels, request_loss=request_loss, **kwargs)

    def run_model(self, x, *args, **kwargs):
        return self.net.forward(x)


class ELANHandler(BaseModel):
    """
    Main handler for ELAN.
    """

    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, in_features=3, perceptual=None,
                 scheduler=None, scheduler_params=None, **kwargs):
        super(ELANHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                          **kwargs)
        self.net = ELAN(scale=scale, colors=in_features, **kwargs)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)
        self.model_name = 'elan'
        if scheduler == 'multi_step_lr':
            self.end_epoch_scheduler = True
        else:
            self.end_epoch_scheduler = False

    def run_train(self, x, y, tag=None, mask=None, keep_on_device=False, *args, **kwargs):
        self.net.train()
        x, y = x.to(device=self.device), y.to(device=self.device)

        out = self.run_model(x)
        loss = self.criterion(out, y)

        self.standard_update(loss, scheduler_skip=self.end_epoch_scheduler)

        return loss.detach().cpu().numpy(), out.detach().cpu()

    def epoch_end_calls(self):
        if self.learning_rate_scheduler is not None and isinstance(self.learning_rate_scheduler,
                                                                   torch.optim.lr_scheduler.MultiStepLR):
            self.learning_rate_scheduler.step()
