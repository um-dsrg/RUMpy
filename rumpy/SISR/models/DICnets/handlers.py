from .DIC_architecture import *
from rumpy.SISR.models import *
import pickle
import os
import re
import torch
from torch import optim as optim

from rumpy.shared_framework.models.base_architecture import BaseModel


class DICHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, hr_data_loc=None, **kwargs):
        super(DICHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                          hr_data_loc=hr_data_loc, **kwargs)
        self.net = DIC(device=device, scale=scale, **kwargs)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.define_optimizer(lr=lr)
        self.model_name = 'dic'
        self.criterion_pixel = nn.L1Loss()
        self.criterion_alignment = nn.MSELoss()
        self.learning_rate_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,  # TODO: can parametrize
                                                                      milestones=[10000, 20000, 40000, 80000],
                                                                      gamma=0.5)
        if not eval_mode:  # TODO: find a better fix
            self.landmarks = pickle.load(open(os.path.join(hr_data_loc, 'landmarks.pkl'), 'rb'))
        self.steps = 0
        self.hg_require_grad = True
        self.scale = scale

    def mod_HG_grad(self, requires_grad=False):
        # taken from https://github.com/Maclory/Deep-Iterative-Collaboration
        if self.hg_require_grad != requires_grad:
            if isinstance(self.net, nn.DataParallel):
                for p in self.net.module.HG.parameters():
                    p.requires_grad = requires_grad
            else:
                for p in self.net.HG.parameters():
                    p.requires_grad = requires_grad

    @staticmethod
    def _generate_one_heatmap(size, landmark, sigma):
        # taken from https://github.com/Maclory/Deep-Iterative-Collaboration
        w, h = size
        x_range = np.arange(start=0, stop=w, dtype=int)
        y_range = np.arange(start=0, stop=h, dtype=int)
        xx, yy = np.meshgrid(x_range, y_range)
        d2 = (xx - landmark[0])**2 + (yy - landmark[1])**2
        exponent = d2 / 2.0 / sigma / sigma
        heatmap = np.exp(-exponent)
        return heatmap

    def get_landmarks(self, image_names, hr_shape):
        h_size = int(hr_shape[2] // (self.scale / 2))
        w_size = int(hr_shape[3] // (self.scale / 2))
        batch_landmarks = torch.zeros((len(image_names), 68, h_size, w_size))  # TODO: parametrize
        for index, image_name in enumerate(image_names):
            landmarks = self.landmarks[re.sub('_(.*?)\.', '.', image_name)]
            landmark_resized = landmarks/(self.scale/2)
            heatmap_list = []
            for coord_index in range(landmark_resized.shape[0]):
                heatmap_list.append(
                    self._generate_one_heatmap((h_size, w_size),
                                               landmark_resized[coord_index, :], 1))
            gt_heatmap = np.stack(heatmap_list, axis=0)
            gt_heatmap = torch.from_numpy(np.ascontiguousarray(gt_heatmap))
            batch_landmarks[index, ...] = gt_heatmap

        return batch_landmarks

    def run_train(self, x, y, image_names=None, masks=None, *args, **kwargs):

        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        batch_landmarks = self.get_landmarks(image_names, y.size()).to(device=self.device)
        self.steps += 1
        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)
        x, y = x.to(device=self.device), y.to(device=self.device)
        out_list, heatmap_list = self.run_model(x)  # run data through model
        loss_pix = 0.0
        loss_align = 0.0
        for step, SR in enumerate(out_list):
            loss_pix += self.criterion_pixel(SR, y)
            loss_align += 0.1 * self.criterion_alignment(heatmap_list[step], batch_landmarks)

        loss = loss_pix + loss_align

        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters
        self.learning_rate_scheduler.step()

        if self.steps >= 2000000:  # TODO: can parametrize
            if self.hg_require_grad is not True:
                # print("Releasing HG gradients")
                self.mod_HG_grad(requires_grad=True)
        else:
            if self.hg_require_grad is not False:
                # print("Fixing HG gradients")
                self.mod_HG_grad(requires_grad=False)

        return {
            'pix_loss': loss_pix.item(),
            'align_loss': loss_align.item(),
            'full_loss': loss.item()}

    def run_eval(self, x, y=None, request_loss=False, image_names=None, *args, **kwargs):

        self.net.eval()  # sets the system to validation mode

        with torch.no_grad():
            x = x.to(device=self.device)
            out_list, heatmap_list = self.run_model(x)  # forward the data in the model
            if request_loss:
                y = y.to(device=self.device)
                batch_landmarks = self.get_landmarks(image_names, y.size()).to(device=self.device)
                loss_pix = 0.0
                loss_align = 0.0
                for step, SR in enumerate(out_list):  # TODO: implement method to output different types of losses at eval time
                    loss_pix += self.criterion_pixel(SR, y)
                    loss_align += 0.1 * self.criterion_alignment(heatmap_list[step], batch_landmarks)
                loss = loss_pix + loss_align
                loss = loss.cpu().data.numpy()  # compute loss
            else:
                loss = None

        return out_list[-1].cpu().data, loss
