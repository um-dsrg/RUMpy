from .architectures import *
from rumpy.shared_framework.models.base_architecture import BaseModel
from rumpy.SISR.models.attention_manipulators import QModel


class SPARNetHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, hr_data_loc=None,
                 scheduler=None, scheduler_params=None, perceptual=None, **kwargs):
        super(SPARNetHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                             hr_data_loc=hr_data_loc, **kwargs)
        self.net = SPARNet(**kwargs)
        self.colorspace = 'rgb'
        self.im_input = 'interp'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)
        self.model_name = 'sparnet'
        self.criterion = nn.L1Loss()
        self.scale = scale


class QSPARNetHandler(QModel):
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, hr_data_loc=None,
                 scheduler=None, scheduler_params=None, perceptual=None, **kwargs):
        super(QSPARNetHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                              hr_data_loc=hr_data_loc, **kwargs)

        self.net = QSPARNet(metadata_count=self.num_metadata, **kwargs)
        self.colorspace = 'rgb'
        self.im_input = 'interp'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)
        self.model_name = 'qsparnet'
        self.criterion = nn.L1Loss()
        self.scale = scale


