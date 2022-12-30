import os

import numpy as np
import torch
from torch import nn
from rumpy.shared_framework.models.base_architecture import BaseModel


class BaseBSRGANModel(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, **kwargs):
        """
        Main base handler for BSRGAN, to be inherited by other handlers.
        """
        super(BaseBSRGANModel, self).__init__(device=device,
                                              model_save_dir=model_save_dir,
                                              eval_mode=eval_mode,
                                              **kwargs)

        self.lambda_adv = None
        self.lambda_pixel = None
        self.lambda_vgg = None

        self.vgg_feature_layers = None
        self.vgg_layer_weights = None

        self.pretrain_epochs = None

        if not self.eval_mode:
            self.discriminator = None
            self.vgg_extractor = None

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
