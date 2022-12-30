from rumpy.shared_framework.models.base_architecture import BaseModel
from rumpy.SISR.models.waveletnets.architectures import *
from rumpy.SISR.models.feature_extractors.handlers import perceptual_loss_mechanism
from rumpy.SISR.models import *


class WaveletSRNetHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, **kwargs):
        super(WaveletSRNetHandler, self).__init__(device=device, model_save_dir=model_save_dir,
                                                  eval_mode=eval_mode, **kwargs)
        self.net = WaveletSRNet(scale=scale)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.define_optimizer(lr=lr)
        self.model_name = 'waveletsrnet'
        self.criterion = nn.L1Loss()  # TODO: won't reflect actual training loss this way (need to change, along with other losses)
        # self.criterion_lr = nn.MSELoss()
        # self.criterion_sr = nn.MSELoss()
        # self.criterion_image = nn.MSELoss()

        self.wavelet_dec = WaveletTransform(scale=scale, dec=True).to(device=self.device)

    def run_train(self, x, y, image_names=None, **kwargs):

        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)

        x, y = x.to(device=self.device), y.to(device=self.device)

        target_wavelets = self.wavelet_dec(y)

        wavelets_lr = target_wavelets[:, 0:3, :, :]
        wavelets_sr = target_wavelets[:, 3:, :, :]

        wavelets_predict, out = self.net.forward(x, train=True)

        loss_lr = loss_MSE(wavelets_predict[:, 0:3, :, :], wavelets_lr, size_average=True)
        loss_sr = loss_MSE(wavelets_predict[:, 3:, :, :], wavelets_sr, size_average=True)

        loss_textures = loss_Textures(wavelets_predict[:, 3:, :, :], wavelets_sr)
        loss_img = loss_MSE(out, y)
        loss = loss_sr.mul(0.99) + loss_lr.mul(0.01) + loss_img.mul(0.1) + loss_textures.mul(1)

        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss

        if self.grad_clip is not None:  # gradient clipping
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()  # update network parameters

        loss_package = {}

        for _loss, name in zip((loss, loss_lr, loss_sr, loss_img, loss_textures),
                               ('full_loss', 'wavelet_lr_loss', 'wavelet_hr_loss', 'img_loss', 'texture_loss')):
            loss_package[name] = _loss.cpu().data.numpy()

        return loss_package


class WaveletSRGANHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, discriminator_lr=1e-4, training_switch=10,
                 scale=8, **kwargs):
        super(WaveletSRGANHandler, self).__init__(device=device, model_save_dir=model_save_dir,
                                                  eval_mode=eval_mode, **kwargs)
        self.net = WaveletSRNet(scale=scale)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.define_optimizer(lr=lr)
        self.model_name = 'waveletsrgan'
        self.training_switch = training_switch
        self.wavelet_dec = WaveletTransform(scale=math.pow(2, scale), dec=True).to(device=self.device)

        if not self.eval_mode:
            self.criterion = nn.L1Loss()  # TODO: won't reflect actual training loss this way (need to change)
        if not self.eval_mode:
            self.discriminator = WaveletDiscriminator(scale=scale)
            self.discriminator.to(self.device)

            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)
            if self.device == torch.device('cpu'):
                dev = torch.device('cpu')
            else:
                dev = 'cuda:%d' % self.device
            self.identity_extractor = perceptual_loss_mechanism('lightcnn', device=dev)

    def set_multi_gpu(self, device_ids=None):
        self.net = nn.DataParallel(self.net, device_ids=device_ids)
        self.discriminator = nn.DataParallel(module=self.discriminator, device_ids=device_ids)

    def run_train(self, x, y, image_names=None, **kwargs):

        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)
        self.discriminator.train()
        self.identity_extractor.eval()

        x, y = x.to(device=self.device), y.to(device=self.device)

        target_wavelets = self.wavelet_dec(y)

        wavelets_lr = target_wavelets[:, 0:3, :, :]
        wavelets_sr = target_wavelets[:, 3:, :, :]

        wavelets_predict, out = self.net.forward(x, train=True)

        loss_lr = loss_MSE(wavelets_predict[:, 0:3, :, :], wavelets_lr, size_average=True)
        loss_sr = loss_MSE(wavelets_predict[:, 3:, :, :], wavelets_sr, size_average=True)

        if self.curr_epoch < self.training_switch:
            loss = loss_sr.mul(0.99) + loss_lr.mul(0.01)
        else:
            fake_readings = self.discriminator(wavelets_predict)  # TODO: check grad is ok here...
            const = torch.tensor(1, requires_grad=False).expand_as(fake_readings).to(self.device)
            adv_loss = ((fake_readings - const) ** 2).sum()/(2*(fake_readings.size()[2] + fake_readings.size()[3]))
            feat_orig = self.identity_extractor.module.extract_features(self.identity_extractor.module.preprocess(y))  # TODO: really needs to be reduced in size
            feat_pred = self.identity_extractor.module.extract_features(self.identity_extractor.module.preprocess(out))
            id_loss = identity_loss(feat_orig, feat_pred)
            loss = loss_sr.mul(0.99) + loss_lr.mul(0.01) + id_loss.mul(10) + adv_loss.mul(10)

        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss

        if self.grad_clip is not None:  # gradient clipping
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()  # update network parameters

        if self.curr_epoch >= self.training_switch:  # Discriminator training
            fake_readings = self.discriminator(wavelets_predict.detach())
            real_readings = self.discriminator(target_wavelets)
            const = torch.tensor(1, requires_grad=False).expand_as(fake_readings).to(self.device)
            reading_size = real_readings.size()[2] + real_readings.size()[3]
            dis_loss = (((real_readings - const) ** 2).sum()/(2*reading_size)) + ((fake_readings ** 2).sum()/(2*reading_size))
            self.optimizer_D.zero_grad()
            dis_loss.backward()
            self.optimizer_D.step()

        loss_package = {}

        if self.curr_epoch < self.training_switch:
            id_loss = torch.tensor(0)
            adv_loss = torch.tensor(0)
            dis_loss = torch.tensor(0)
        for _loss, name in zip((loss, loss_lr, loss_sr, id_loss, adv_loss, dis_loss),
                               ('full_loss', 'wavelet_lr_loss', 'wavelet_hr_loss', 'id_loss', 'adv_loss', 'discrim_loss')):
            loss_package[name] = _loss.cpu().data.numpy()

        return loss_package

# TODO: add full loss return for eval iteration too
