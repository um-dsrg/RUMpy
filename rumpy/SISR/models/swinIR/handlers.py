###UNDER CONSTRUCTION###
from rumpy.shared_framework.models.base_architecture import BaseModel
from rumpy.SISR.models.swinIR.architectures import SwinIR


class SwinIRHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, scale=4,
                 **kwargs):
        super(SwinIRHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                            **kwargs)

        self.net = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8,
                          img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                          num_heads=[6, 6, 6, 6, 6, 6],
                          mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')

        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
