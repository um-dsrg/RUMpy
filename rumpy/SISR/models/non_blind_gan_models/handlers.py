from rumpy.shared_framework.models.base_architecture import BaseModel
from rumpy.SISR.models.non_blind_gan_models import BaseBSRGANModel
from rumpy.SISR.models.non_blind_gan_models.generators import RRDBNet
from rumpy.SISR.models.non_blind_gan_models.generators_bsrgan import RRDBNet as RRDBNetBSRGAN
from rumpy.SISR.models.non_blind_gan_models.discriminators import *
from rumpy.SISR.models.feature_extractors.handlers import perceptual_loss_mechanism
from rumpy.SISR.models.implicit_blind_sr.fssr_modules.models_esrganfs import RRDBNet as RRDBfssr
from rumpy.sr_tools.stats import load_statistics
from rumpy.sr_tools.helper_functions import standard_metric_epoch_selection

import numpy as np
import toml
import os
import torch


class ESRGANHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, scale=4,
                 pretrain_lr=2e-4, main_lr=1e-4, discriminator_lr=1e-4,
                 lambda_adv=5e-3, lambda_pixel=1e-2,
                 pretrain_epochs=1000,
                 pre_train_optimizer_params=None,
                 main_optimizer_params=None,
                 discriminator_optimizer_params=None,
                 pre_train_scheduler=None,
                 pre_train_scheduler_params=None,
                 main_scheduler=None,
                 main_scheduler_params=None,
                 fssr_model_structure=False,
                 **kwargs):
        """
        Main handler for ESRGAN.  Please note eval loss is always L1loss, and does not consider extra types of losses.
        """
        super(ESRGANHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                            **kwargs)
        if not fssr_model_structure:
            self.net = RRDBNet(scale=scale)
        else:
            if scale != 4:
                raise RuntimeError('FSSR model can only be used with scale factor 4.')
            self.net = RRDBfssr()
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()

        self.lambda_adv = lambda_adv
        self.lambda_pixel = lambda_pixel
        self.pretrain_epochs = pretrain_epochs

        # Specific Optimizer, Discriminator and Scheduler Config (only for training)
        self.optimizer = {}
        self.learning_rate_scheduler = {}

        if not self.eval_mode:
            if pretrain_epochs != 0:
                self.optimizer['pre_train_optimizer'] = self.define_optimizer(self.net.parameters(),
                                                                              lr=pretrain_lr,
                                                                              optimizer_params=pre_train_optimizer_params)

            self.optimizer['main_optimizer'] = self.define_optimizer(self.net.parameters(),
                                                                     lr=main_lr, optimizer_params=main_optimizer_params)

            if pre_train_scheduler is not None:
                self.learning_rate_scheduler['pre_train_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['pre_train_optimizer'],
                    scheduler=pre_train_scheduler,
                    scheduler_params=pre_train_scheduler_params)

            if main_scheduler is not None:
                self.learning_rate_scheduler['main_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['main_optimizer'],
                    scheduler=main_scheduler,
                    scheduler_params=main_scheduler_params)

            # discriminator model
            self.discriminator = VGGStyleDiscriminator128()
            self.discriminator.to(self.device)
            self.optimizer['discrim_optimizer'] = self.define_optimizer(self.discriminator.parameters(),
                                                                        lr=discriminator_lr,
                                                                        optimizer_params=discriminator_optimizer_params)
            if main_scheduler is not None:  # same scheduler used for discriminator as for main generator
                self.learning_rate_scheduler['discrim_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['discrim_optimizer'],
                    scheduler=main_scheduler,
                    scheduler_params=main_scheduler_params)

            # perceptual loss mechanism
            self.vgg_extractor = perceptual_loss_mechanism('vgg', mode='p_loss', device=device)
            self.vgg_extractor.to(self.device)
            self.vgg_extractor.eval()

            # additional error criteria
            self.criterion_GAN = nn.BCEWithLogitsLoss()
            self.criterion_content = nn.L1Loss()

    def set_multi_gpu(self, device_ids=None):
        self.net = nn.DataParallel(self.net, device_ids=device_ids)
        if not self.eval_mode:
            self.discriminator = nn.DataParallel(module=self.discriminator, device_ids=device_ids)
            self.vgg_extractor = nn.DataParallel(module=self.vgg_extractor, device_ids=device_ids)
            self.vgg_extractor.eval()

    def pre_train_update(self, loss):
        self.optimizer['pre_train_optimizer'].zero_grad()
        loss.backward()
        self.optimizer['pre_train_optimizer'].step()  # update network parameters
        self.learning_rate_scheduler['pre_train_scheduler'].step()
        loss_package = {}
        for _loss, name in zip((loss, loss, torch.tensor(0), torch.tensor(0), torch.tensor(0)),
                               ('train-loss', 'l1-loss', 'gan-loss', 'vgg-loss', 'discriminator-loss')):
            loss_package[name] = _loss.cpu().data.numpy()
        return loss_package

    def generator_update(self, gen_image, ref_image):
        for p in self.discriminator.parameters():  # TODO: is this really needed?
            p.requires_grad = False

        # L1 loss
        loss = self.criterion(gen_image, ref_image)

        # Content loss
        gen_features = self.vgg_extractor(gen_image)
        real_features = self.vgg_extractor(ref_image).detach()
        loss_content = self.criterion_content(gen_features, real_features)

        # Extract validity predictions from discriminator
        pred_real = self.discriminator(ref_image).detach()
        pred_fake = self.discriminator(gen_image)

        # Adversarial loss (relativistic average GAN)
        valid = pred_real.new_ones(pred_real.size()) * 1.0
        fake = pred_fake.new_ones(pred_fake.size()) * 0.0
        loss_GAN_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), fake)
        loss_GAN_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
        loss_GAN = (loss_GAN_fake + loss_GAN_real) / 2

        # Total generator loss
        loss_G = loss_content + self.lambda_adv * loss_GAN + self.lambda_pixel * loss

        self.optimizer['main_optimizer'].zero_grad()
        loss_G.backward()
        self.optimizer['main_optimizer'].step()
        self.learning_rate_scheduler['main_scheduler'].step()
        return loss_G, loss_content, loss_GAN, loss

    def discriminator_update(self, gen_image, ref_image):
        for p in self.discriminator.parameters():  # TODO: is this required?
            p.requires_grad = True

        self.optimizer['discrim_optimizer'].zero_grad()

        # repetition of these processes due to detachment of gradients required
        pred_real = self.discriminator(ref_image)
        pred_fake = self.discriminator(gen_image).detach()

        # Adversarial loss for real and fake images (relativistic average GAN)
        valid = pred_real.new_ones(pred_real.size()) * 1.0
        fake = pred_fake.new_ones(pred_fake.size()) * 0.0
        loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        self.optimizer['discrim_optimizer'].step()
        self.learning_rate_scheduler['discrim_scheduler'].step()

        return loss_D

    def run_train(self, x, y, *args, **kwargs):
        """
        Runs one training iteration (pre-train or GAN) through a data batch
        :param x: input
        :param y: target
        :return: calculated loss pre-backprop, output image
        """
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)
        self.discriminator.train()

        x, y = x.to(device=self.device), y.to(device=self.device)

        if self.curr_epoch < self.pretrain_epochs:  # L1 pre-training
            out = self.net.forward(x)  # run data through model
            loss = self.criterion(out, y)  # compute L1 loss
            loss_package = self.pre_train_update(loss)
            return loss_package, out.detach().cpu()
        else:
            out = self.net.forward(x)  # run data through model
            loss_G, loss_content, loss_GAN, loss = self.generator_update(out, y)
            loss_D = self.discriminator_update(out, y)

        loss_package = {}

        for _loss, name in zip((loss_G, loss, loss_GAN, loss_content, loss_D),
                               ('train-loss', 'l1-loss', 'gan-loss', 'vgg-loss', 'discriminator-loss')):
            loss_package[name] = _loss.cpu().data.numpy()

        return loss_package, out.detach().cpu()

    def print_parameters(self, verbose=False):  # TODO: see if this .net focus can be used for all models
        """
        Reports how many learnable parameters are available in the model, and where they are distributed.
        :return: None
        """
        if verbose:
            print('----------------------------')
            print('Parameter names:')
        total_num_parameters = 0
        for name, value in self.net.named_parameters():
            if verbose:
                print(name, value.shape)
            total_num_parameters += np.prod(value.shape)
        if verbose:
            print('Total number of trainable parameters:', total_num_parameters)
            print('----------------------------')
        return total_num_parameters

    def extra_diagnostics(self):
        if not self.eval_mode:
            discrim_params = 0
            for name, value in self.discriminator.named_parameters():
                discrim_params += np.prod(value.shape)
            print('Discriminator loaded with the %s architecture (%d parameters).' % (
                self.discriminator.__class__.__name__, discrim_params))

    def get_learning_rate(self):  # TODO: this could also be generalised for all multi discriminator models
        lrs = {}
        for key, optimizer in self.optimizer.items():
            lrs['%s_learning_rate' % key] = optimizer.param_groups[0]['lr']
        return lrs

    @staticmethod
    def best_model_selection_criteria(log_dir=None, log_file='summary.csv', model_metadata=None,
                                      stats=None, stats_dir=None, base_metric='val-PSNR'):
        """
        GAN models typically have a pre-training phase.  This function ignores this phase when
        finding the best epoch based on the selected metric.
        """

        if stats_dir and stats is None:
            stats = load_statistics(log_dir, log_file, config='pd')  # loads model training stats if not provided

        if not model_metadata:
            model_metadata = toml.load(os.path.join(os.path.dirname(log_dir), 'config.toml'))['model']

        cutoff_epoch = model_metadata['pretrain_epochs']

        load_epoch = standard_metric_epoch_selection(base_metric, stats[cutoff_epoch:])
        return load_epoch


class BSRGANHandler(BaseBSRGANModel):
    def __init__(self, device, model_save_dir, eval_mode=False, scale=4,
                 pretrain_lr=1e-4,
                 main_lr=5e-5,
                 discriminator_lr=5e-5,
                 lambda_adv=1.0,
                 lambda_pixel=1.0,
                 lambda_vgg=1.0,
                 pretrain_epochs=100,
                 vgg_feature_layers=[2, 7, 16, 25, 34],
                 vgg_layer_weights=[0.1, 0.1, 1.0, 1.0, 1.0],
                 # optimizers and schedulers
                 pre_train_optimizer_params=None,
                 main_optimizer_params=None,
                 discriminator_optimizer_params=None,
                 pre_train_scheduler=None,
                 pre_train_scheduler_params=None,
                 main_scheduler=None,
                 main_scheduler_params=None,
                 **kwargs):
        """
        Main handler for BSRGAN.
        """
        super(BSRGANHandler, self).__init__(device=device,
                                            model_save_dir=model_save_dir,
                                            eval_mode=eval_mode,
                                            **kwargs)

        self.net = RRDBNetBSRGAN()
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()

        self.lambda_adv = lambda_adv
        self.lambda_pixel = lambda_pixel
        self.lambda_vgg = lambda_vgg

        self.vgg_feature_layers = vgg_feature_layers
        self.vgg_layer_weights = vgg_layer_weights

        # Specific Optimizer, Discriminator and Scheduler Config (only for training)
        self.optimizer = {}
        self.learning_rate_scheduler = {}
        self.pretrain_epochs = pretrain_epochs

        if not self.eval_mode:
            if pretrain_epochs != 0:
                self.optimizer['pre_train_optimizer'] = self.define_optimizer(self.net.parameters(),
                                                                              lr=pretrain_lr,
                                                                              optimizer_params=pre_train_optimizer_params)

            self.optimizer['main_optimizer'] = self.define_optimizer(self.net.parameters(),
                                                                     lr=main_lr,
                                                                     optimizer_params=main_optimizer_params)

            if pre_train_scheduler is not None and pretrain_epochs != 0:
                self.learning_rate_scheduler['pre_train_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['pre_train_optimizer'],
                    scheduler=pre_train_scheduler,
                    scheduler_params=pre_train_scheduler_params)

            if main_scheduler is not None:
                self.learning_rate_scheduler['main_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['main_optimizer'],
                    scheduler=main_scheduler,
                    scheduler_params=main_scheduler_params)

            self.discriminator = Discriminator_UNet()
            self.discriminator.to(self.device)

            self.optimizer['discrim_optimizer'] = self.define_optimizer(self.discriminator.parameters(),
                                                                        lr=discriminator_lr,
                                                                        optimizer_params=discriminator_optimizer_params)

            if main_scheduler is not None:  # same scheduler used for discriminator as for main generator
                self.learning_rate_scheduler['discrim_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['discrim_optimizer'],
                    scheduler=main_scheduler,
                    scheduler_params=main_scheduler_params)

            # perceptual loss mechanism
            self.vgg_extractor = perceptual_loss_mechanism('vgg', mode=vgg_feature_layers, device=device)
            self.vgg_extractor.to(self.device)
            self.vgg_extractor.eval()

            # additional error criteria
            self.criterion_GAN = nn.MSELoss()
            self.criterion_content = nn.L1Loss()

    def perceptual_loss(self, gen_features, ref_features):
        loss = 0.0
        if isinstance(gen_features, list):
            n = len(gen_features)
            for i in range(n):
                loss += self.vgg_layer_weights[i] * self.criterion_content(gen_features[i], ref_features[i])
        else:
            loss += self.criterion_content(gen_features, ref_features.detach())
        return loss

    def pre_train_update(self, loss):
        self.optimizer['pre_train_optimizer'].zero_grad()
        loss.backward()
        self.optimizer['pre_train_optimizer'].step()  # update network parameters

        self.learning_rate_scheduler['pre_train_scheduler'].step()

    def generator_update(self, gen_image, ref_image):
        for p in self.discriminator.parameters():  # TODO: is this really needed?
            p.requires_grad = False

        # L1 loss
        loss = self.criterion(gen_image, ref_image)

        # Content loss
        gen_features = self.vgg_extractor(gen_image)
        real_features = self.vgg_extractor(ref_image)
        loss_content = self.perceptual_loss(gen_features, real_features)

        # Extract validity predictions from discriminator
        pred_fake = self.discriminator(gen_image)

        # Adversarial loss (relativistic average GAN)
        target_real = pred_fake.new_ones(pred_fake.size()) * 1.0
        loss_GAN = self.criterion_GAN(pred_fake, target_real)

        # Total generator loss
        loss_G = (loss_content * self.lambda_vgg) + (self.lambda_adv * loss_GAN) + (self.lambda_pixel * loss)

        self.optimizer['main_optimizer'].zero_grad()
        loss_G.backward()
        self.optimizer['main_optimizer'].step()

        self.learning_rate_scheduler['main_scheduler'].step()

        return loss_G, loss_content, loss_GAN, loss

    def discriminator_update(self, gen_image, ref_image):
        for p in self.discriminator.parameters():  # TODO: is this required?
            p.requires_grad = True

        self.optimizer['discrim_optimizer'].zero_grad()

        pred_real = self.discriminator(ref_image)
        target_real = pred_real.new_ones(pred_real.size()) * 1.0
        loss_real = self.criterion_GAN(pred_real, target_real)
        loss_real.backward()

        # Detach to avoid errors with re-running data through graph
        pred_fake = self.discriminator(gen_image.detach().clone())
        target_fake = pred_fake.new_ones(pred_fake.size()) * 0.0
        loss_fake = self.criterion_GAN(pred_fake, target_fake)
        loss_fake.backward()

        self.optimizer['discrim_optimizer'].step()
        self.learning_rate_scheduler['discrim_scheduler'].step()

        return loss_real, loss_fake

    def run_train(self, x, y, *args, **kwargs):
        """
        Runs one training iteration (pre-train or GAN) through a data batch
        :param x: input
        :param y: target
        :return: calculated loss pre-backprop, output image
        """
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)
        self.discriminator.train()

        x, y = x.to(device=self.device), y.to(device=self.device)

        if self.curr_epoch < self.pretrain_epochs:  # L1 pre-training
            out = self.net.forward(x)  # run data through model
            loss = self.criterion(out, y)  # compute L1 loss
            self.pre_train_update(loss)

            loss_G = loss
            loss_content = torch.tensor(0.0, device=self.device)
            loss_GAN = torch.tensor(0.0, device=self.device)
            loss_real = torch.tensor(0.0, device=self.device)
            loss_fake = torch.tensor(0.0, device=self.device)
        else:
            out = self.net.forward(x)  # run data through model
            loss_G, loss_content, loss_GAN, loss = self.generator_update(out, y)
            loss_real, loss_fake = self.discriminator_update(out, y)

        loss_package = {}

        for _loss, name in zip((loss_G, loss, loss_GAN, loss_content, loss_real, loss_fake),
                               ('train-loss', 'l1-loss', 'gan-loss', 'vgg-loss', 'd-loss-real', 'd-loss-fake')):
            loss_package[name] = _loss.cpu().data.numpy()

        return loss_package, out.detach().cpu()

    def extra_diagnostics(self):
        if not self.eval_mode:
            models = [self.net, self.discriminator, self.vgg_extractor]
            model_names = ['Generator', 'Discriminator', 'VGG Extractor']
            self.print_parameters_model_list(models, model_names)

    def get_learning_rate(self):  # TODO: this could also be generalised for all multi discriminator models
        lrs = {}
        for key, optimizer in self.optimizer.items():
            lrs['%s_learning_rate' % key] = optimizer.param_groups[0]['lr']
        return lrs


class RealESRGANHandler(BaseBSRGANModel):
    def __init__(self, device, model_save_dir, eval_mode=False, scale=4,
                 pretrain_lr=2e-4,
                 main_lr=1e-4,
                 discriminator_lr=1e-4,
                 lambda_adv=0.1,
                 lambda_pixel=1.0,
                 lambda_vgg=1.0,
                 pretrain_epochs=100,
                 vgg_feature_layers=[2, 7, 16, 25, 34],
                 vgg_layer_weights=[0.1, 0.1, 1.0, 1.0, 1.0],
                 # optimizers and schedulers
                 pre_train_optimizer_params=None,
                 main_optimizer_params=None,
                 discriminator_optimizer_params=None,
                 pre_train_scheduler=None,
                 pre_train_scheduler_params=None,
                 main_scheduler=None,
                 main_scheduler_params=None,
                 **kwargs):
        """
        Main handler for RealESRGAN - very similar to BSRGAN with some small changes.
        Changes include:
        - the naming of model parameters
        - the type of GAN loss
        """
        super(RealESRGANHandler, self).__init__(device=device,
                                                model_save_dir=model_save_dir,
                                                eval_mode=eval_mode,
                                                **kwargs)
        self.net = RRDBNet(scale=scale)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()

        self.lambda_adv = lambda_adv
        self.lambda_pixel = lambda_pixel
        self.lambda_vgg = lambda_vgg

        self.vgg_feature_layers = vgg_feature_layers
        self.vgg_layer_weights = vgg_layer_weights

        # Specific Optimizer, Discriminator and Scheduler Config (only for training)
        self.optimizer = {}
        self.learning_rate_scheduler = {}
        self.pretrain_epochs = pretrain_epochs

        if not self.eval_mode:
            if pretrain_epochs != 0:
                self.optimizer['pre_train_optimizer'] = self.define_optimizer(self.net.parameters(),
                                                                              lr=pretrain_lr,
                                                                              optimizer_params=pre_train_optimizer_params)

            self.optimizer['main_optimizer'] = self.define_optimizer(self.net.parameters(),
                                                                     lr=main_lr,
                                                                     optimizer_params=main_optimizer_params)

            if pre_train_scheduler is not None and pretrain_epochs != 0:
                self.learning_rate_scheduler['pre_train_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['pre_train_optimizer'],
                    scheduler=pre_train_scheduler,
                    scheduler_params=pre_train_scheduler_params)

            if main_scheduler is not None:
                self.learning_rate_scheduler['main_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['main_optimizer'],
                    scheduler=main_scheduler,
                    scheduler_params=main_scheduler_params)

            self.discriminator = UNetDiscriminatorSN()
            self.discriminator.to(self.device)

            self.optimizer['discrim_optimizer'] = self.define_optimizer(self.discriminator.parameters(),
                                                                        lr=discriminator_lr,
                                                                        optimizer_params=discriminator_optimizer_params)

            if main_scheduler is not None:  # same scheduler used for discriminator as for main generator
                self.learning_rate_scheduler['discrim_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['discrim_optimizer'],
                    scheduler=main_scheduler,
                    scheduler_params=main_scheduler_params)

            # perceptual loss mechanism
            self.vgg_extractor = perceptual_loss_mechanism('vgg', mode=vgg_feature_layers, device=device)
            self.vgg_extractor.to(self.device)
            self.vgg_extractor.eval()

            # additional error criteria
            self.criterion_GAN = nn.BCEWithLogitsLoss()
            self.criterion_content = nn.L1Loss()
