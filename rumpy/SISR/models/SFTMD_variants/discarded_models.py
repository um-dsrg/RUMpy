import numpy as np
import torch
from torch import nn as nn

from rumpy.shared_framework.models.base_architectures import BaseModel
from rumpy.SISR.models.SFTMD_variants.architectures import SFTMD


class SFTMDCascadeHandler(BaseModel):  # very hacky network, TODO: currently non-functional, needs to be edited to work with new system
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, normalize=True, blur_kernels=False,
                 blur_kernels_only=False, metadata=None,
                 qpi_cutoff=24, **kwargs):
        super(SFTMDCascadeHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                                  **kwargs)
        self.net = SFTMDCascade(**kwargs)
        self.colorspace = 'augmented_rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.define_optimizer(lr=lr)
        self.model_name = 'sftmdcascade'
        self.criterion = nn.L1Loss()

        # if lr_data_loc is None:
        #     raise Exception('Location of compression parameters needs to be specified for this network to run.')

        # if blur_kernels:
        #     kernel_loc = os.path.join(lr_data_loc, 'kernel_maps.npy')
        # else:
        #     kernel_loc = None

        # self.augmentation_list, min_aug, max_aug = read_augmentation_list(os.path.join(lr_data_loc, 'qpi_slices.csv'),
        #                                                                   normalize=normalize, blur_kernels=kernel_loc,
        #                                                                   no_qpi=blur_kernels_only,
        #                                                                   metadata=metadata,
        #                                                                   attributes_loc=data_attributes)
        # if lr_eval_data_loc is not None:
        #     eval_list, _, _ = read_augmentation_list(os.path.join(lr_eval_data_loc, 'qpi_slices.csv'),
        #                                              normalize=normalize, attributes_loc=data_attributes,
        #                                              metadata=metadata)
        #     self.augmentation_list = {**self.augmentation_list, **eval_list}

        # self.qpi_cutoff = (qpi_cutoff - min_aug) / (max_aug - min_aug)

    def generate_channels(self, x, image_names):

        num_channels = np.array(self.augmentation_list[image_names[0]]).size
        extra_channels = torch.ones(x.size(0), num_channels, *x.size()[2:])
        qpi_vals = []
        for index, name in enumerate(image_names):
            if num_channels == 1:
                extra_channels[index, ...] = extra_channels[index, ...] * self.augmentation_list[name]
                qpi_vals.append(self.augmentation_list[name])
            else:
                extra_channels[index, ...] = torch.from_numpy(np.expand_dims(
                    np.expand_dims(self.augmentation_list[name], axis=-1), axis=-1).repeat(x.size()[2], 1).repeat(
                    x.size()[3], 2))

        return extra_channels, qpi_vals

    def run_train(self, x, y, y_med=None, image_names=None, masks=None, *args, **kwargs):

        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)

        extra_channels, qpi_vals = self.generate_channels(x, image_names)
        extra_channels = extra_channels.to(self.device)
        if qpi_vals[0] > self.qpi_cutoff:
            full_net = True
        else:
            full_net = False

        x, y, y_med = x.to(device=self.device), y.to(device=self.device), y_med.to(device=self.device)
        out = self.run_model(x, full_size=full_net, extra_channels=extra_channels)  # run data through model

        if full_net:
            loss_med = self.criterion(out['mid_im'], y_med)  # compute loss
        else:
            loss_med = 0

        if self.face_finder:
            face_mask = self.get_face_mask(y, image_names=image_names, masks=masks).to(device=self.device)
            y = y * face_mask
            out['final_im'] = out['final_im'] * face_mask

        loss_final = self.criterion(out['final_im'], y)   # compute loss

        loss = loss_med + loss_final

        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss

        # self.learning_rate_scheduler.step(epoch=self.current_epoch)  # TODO: Add if scheduler used
        if self.grad_clip is not None:  # gradient clipping
            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()  # update network parameters

        return loss.cpu().data.numpy()

    def run_eval(self, x, y=None, request_loss=False, image_names=None, *args, **kwargs):
        extra_channels, qpi_vals = self.generate_channels(x, image_names)
        extra_channels = extra_channels.to(self.device)
        list_output = []
        for index in range(x.shape[0]):
            full_net = False
            if qpi_vals[index] > self.qpi_cutoff:
                full_net = True
            result, *_ = super().run_eval(x[index, ...].unsqueeze(0), request_loss=False, eval_run=True,
                                         extra_channels=extra_channels[index, ...].unsqueeze(0),
                                         full_size=full_net)
            list_output.append(result)

        output = torch.stack(list_output).squeeze()

        if request_loss:
            if y is None:
                print('Loss cannot be calculated, reference image not provided.')
                loss = None
            else:
                y = y.to(self.device)
                loss = self.criterion(output, y).cpu().data.numpy()
        else:
            loss = None
        return output.cpu().data, loss

    def run_model(self, x, extra_channels=None, full_size=False, eval_run=False, *args, **kwargs):
        return self.net.forward(x, extra_channels=extra_channels, full_size=full_size, eval_run=eval_run)


class SFTMDCascade(SFTMD):
    def __init__(self, in_nc=3, out_nc=3, num_features=64, **kwargs):
        super(SFTMDCascade, self).__init__(in_nc=in_nc, out_nc=out_nc, num_features=num_features, **kwargs)
        self.conv_comp_out = nn.Conv2d(in_channels=num_features, out_channels=out_nc, kernel_size=3, stride=1,
                                       padding=1, bias=True)
        self.conv1_1 = nn.Conv2d(in_nc, num_features, 3, stride=1, padding=1)
        self.relu_conv1_1 = nn.LeakyReLU(0.2)
        self.conv2_2 = nn.Conv2d(num_features, num_features, 3, stride=1, padding=1)
        self.relu_conv2_2 = nn.LeakyReLU(0.2)
        self.conv3_3 = nn.Conv2d(num_features, num_features, 3, stride=1, padding=1)

    def forward(self, x, extra_channels, full_size=True, eval_run=False):

        if full_size:
            fea_bef = self.conv3(self.relu_conv2(self.conv2(self.relu_conv1(self.conv1(x)))))
            fea_in = fea_bef
            for i in range(int(self.num_blocks/2)):
                fea_in = self.__getattr__('SFT-residual' + str(i + 1))(fea_in, extra_channels)
            fea_fin_mid = torch.add(fea_in, fea_bef)
            im_mid = self.conv_comp_out(fea_fin_mid)
        else:
            im_mid = x

        fea_bef = self.conv3_3(self.relu_conv2_2(self.conv2_2(self.relu_conv1_1(self.conv1_1(im_mid)))))
        fea_in = fea_bef

        for i in range(int(self.num_blocks/2), self.num_blocks):
            fea_in = self.__getattr__('SFT-residual' + str(i + 1))(fea_in, extra_channels)

        fea_add = torch.add(fea_in, fea_bef)

        fea = self.upscale(self.conv_mid(self.sft(fea_add, extra_channels)))
        out = self.conv_output(fea)

        if eval_run:
            return out
        else:
            return {'mid_im': im_mid, 'final_im': out}
