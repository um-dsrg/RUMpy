from .architectures import *
from rumpy.SISR.models.attention_manipulators import QModel
from collections import OrderedDict

class SFTMDHandler(QModel):
    def __init__(self, device, eval_mode=False, lr=1e-4, scheduler=None, concat_strategy=False,
                 scheduler_params=None, perceptual=None, q_injection=False, da_injection=False,
                 in_nc=3, optimizer_params=None, **kwargs):
        super(SFTMDHandler, self).__init__(device=device, eval_mode=eval_mode, **kwargs)

        if concat_strategy:  # activating concatenation system
            self.channel_concat = True
            in_nc = self.num_metadata + in_nc

        self.net = SFTMD(input_para=self.num_metadata, q_injection=q_injection, da_injection=da_injection,
                         in_nc=in_nc, **kwargs)

        if q_injection or da_injection:  # converting method of translating metadata
            self.vector_metadata = True
        else:
            self.vector_metadata = False

        self.colorspace = 'augmented_rgb'
        self.im_input = 'unmodified'
        self.activate_device()

        self.training_setup(lr, scheduler, scheduler_params, perceptual, device, optimizer_params=optimizer_params)

        self.model_name = 'sftmd'
        self.criterion = nn.L1Loss()

    def generate_channels(self, x, metadata, metadata_keys, vector_override=False):

        if self.vector_metadata or vector_override:
            # need to generate vector metadata not channel-like metadata if using q-injection or IKC channel generation
            return super().generate_channels(x, metadata, metadata_keys)
        else:
            return super().generate_sft_channels(x, metadata, metadata_keys)

    def legacy_switch(self, state_dict, qrealesrgan_fix=False):
        """
        Function taking care of additional legacy functionality that has been changed for SFTMD models
        :param state_dict: model state dict (used for saving/loading)
        :return: updated state_dict
        """
        new_state_dict = super().legacy_switch(state_dict)
        new_new_state_dict = OrderedDict()
        for k, v in new_state_dict.items():
            if 'sft_branch' in k:
                continue
            elif 'sft_module' in k:
                new_new_state_dict[k] = v
            elif 'sft1' in k or 'sft2' in k:
                new_new_state_dict[k.replace('sft1', 'sft1.sft_module').replace('sft2', 'sft2.sft_module')] = v
            elif k[:4] == 'sft.':
                new_new_state_dict[k.replace('sft.', 'sft.sft_module.')] = v
            else:
                new_new_state_dict[k] = v
        return new_new_state_dict


