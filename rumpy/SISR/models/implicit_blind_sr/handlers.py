from rumpy.shared_framework.models.base_architecture import BaseModel
from rumpy.SISR.models.implicit_blind_sr.fssr_modules.models_dsgan import Generator, Discriminator, FilterLow, FilterHigh
from rumpy.SISR.models.implicit_blind_sr.fssr_modules.loss_functions import GeneratorLoss, discriminator_loss
from rumpy.SISR.models.non_blind_gan_models.handlers import ESRGANHandler

import numpy as np


class ESRGANFSHandler(ESRGANHandler):
    """
    ESRGAN with frequency separation.  Mostly based on ESRGAN implementation
    TODO: can the two esrgans be combined, or could a new base model be made?
    TODO: Ensure a mechanism is in place for loading in pre-trained models
    """
    def __init__(self, use_filters=True, **kwargs):
        super(ESRGANFSHandler, self).__init__(fssr_model_structure=True, **kwargs)

        self.filter_low = FilterLow().to(self.device)
        self.filter_high = FilterHigh().to(self.device)
        self.use_filters = use_filters

    def generator_update(self, gen_image, ref_image):
        for p in self.discriminator.parameters():  # TODO: is this really needed?
            p.requires_grad = False

        # L1 loss
        if self.use_filters:
            loss = self.criterion(self.filter_low(gen_image), self.filter_low(ref_image))
        else:
            loss = self.criterion(gen_image, ref_image)

        # Content loss
        gen_features = self.vgg_extractor(gen_image)
        real_features = self.vgg_extractor(ref_image).detach()
        loss_content = self.criterion_content(gen_features, real_features)

        # Extract validity predictions from discriminator
        if self.use_filters:
            pred_real = self.discriminator(self.filter_high(ref_image)).detach()
            pred_fake = self.discriminator(self.filter_high(gen_image))
        else:
            pred_real = self.discriminator(ref_image).detach()
            pred_fake = self.discriminator(gen_image)

        # Adversarial loss (relativistic average GAN)
        valid = pred_real.new_ones(pred_real.size()) * 1.0
        fake = pred_fake.new_ones(pred_fake.size()) * 0.0
        loss_GAN_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), fake)
        loss_GAN_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
        loss_GAN = (loss_GAN_fake + loss_GAN_real)/2

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

        if self.use_filters:
            pred_real = self.discriminator(self.filter_high(ref_image))
            pred_fake = self.discriminator(self.filter_high(gen_image)).detach()
        else:
            pred_real = self.discriminator(ref_image)  # repetition of these processes due to detachment of gradients required
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


class FSSRDSGANHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False,
                 generator_optimizer_params=None,
                 discriminator_optimizer_params=None,
                 generator_lr=0.0002,
                 discriminator_lr=0.0002,
                 global_scheduler=None,
                 global_scheduler_params=None,
                 ds_epochs=300,
                 decay_epochs=150,
                 **kwargs):
        super(FSSRDSGANHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                               **kwargs)
        self.net = Generator()
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'  # TODO: consider adding another data type for these images....
        self.activate_device()
        self.criterion = GeneratorLoss(device=device)
        self.scale = 1  # model only produces images with the same size as the input
        if global_scheduler == 'custom':
            start_decay = ds_epochs - decay_epochs
            scheduler_function = lambda e: 1.0 if e < start_decay else 1.0 - max(0.0, float(e - start_decay) / decay_epochs)
            global_scheduler_params = {'function': scheduler_function}

        self.optimizer = {}
        self.learning_rate_scheduler = {}

        if not self.eval_mode:
            self.optimizer['generator_optimizer'] = self.define_optimizer(self.net.parameters(),
                                                                          lr=generator_lr,
                                                                          optimizer_params=generator_optimizer_params)

            if global_scheduler is not None:
                self.learning_rate_scheduler['generator_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['generator_optimizer'],
                    scheduler=global_scheduler,
                    scheduler_params=global_scheduler_params)

            # discriminator model
            self.discriminator = Discriminator()
            self.discriminator.to(self.device)
            self.optimizer['discrim_optimizer'] = self.define_optimizer(self.discriminator.parameters(),
                                                                        lr=discriminator_lr,
                                                                        optimizer_params=discriminator_optimizer_params)
            if global_scheduler is not None:  # same scheduler used for discriminator as for main generator
                self.learning_rate_scheduler['discrim_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['discrim_optimizer'],
                    scheduler=global_scheduler,
                    scheduler_params=global_scheduler_params)

    def run_train(self, x, y, *args, **kwargs):

        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)
        self.discriminator.train()

        x, y = x.to(device=self.device), y.to(device=self.device)
        gen_img = self.net.forward(x)  # run data through model

        real_tex = self.discriminator(y)
        fake_tex = self.discriminator(gen_img)

        # update discriminator
        self.optimizer['discrim_optimizer'].zero_grad()
        d_tex_loss = discriminator_loss(real_tex, fake_tex)
        d_tex_loss.backward()
        self.optimizer['discrim_optimizer'].step()

        # Update generator
        self.optimizer['generator_optimizer'].zero_grad()
        g_loss = self.criterion(fake_tex, gen_img, x)
        g_loss.backward()
        self.optimizer['generator_optimizer'].step()

        loss_package = {}

        for _loss, name in zip((g_loss, g_loss, d_tex_loss),
                               ('train-loss', 'generator-loss', 'discriminator-loss')):
            loss_package[name] = _loss.cpu().data.numpy()

        return loss_package, gen_img.detach().cpu()

    def epoch_end_calls(self):  # only call learning rate scheduler once per epoch
        if 'generator_scheduler' in self.learning_rate_scheduler:
            self.learning_rate_scheduler['generator_scheduler'].step()
        if 'discrim_scheduler' in self.learning_rate_scheduler:
            self.learning_rate_scheduler['discrim_scheduler'].step()

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
