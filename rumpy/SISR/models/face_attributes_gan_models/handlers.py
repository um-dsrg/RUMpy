import os
import pandas as pd
import torch
import torch.nn.functional as F
from rumpy.shared_framework.configuration.constants import data_splits
from rumpy.SISR.models import *
from rumpy.SISR.models.attention_manipulators import QModel
from rumpy.SISR.models.face_attributes_gan_models.discriminators import *
from rumpy.SISR.models.face_attributes_gan_models.generators import *
from rumpy.SISR.models.feature_extractors.handlers import perceptual_loss_mechanism
from rumpy.SISR.models.non_blind_gan_models.discriminators import *

class FaceSRAttributesGANHandler(QModel):
    def __init__(self, device, model_save_dir, eval_mode=False, scale=8,
                 generator_lr=1e-3, discriminator_lr=1e-3,
                 min_gen_lr=1e-6, min_dis_lr=1e-6,
                 lambda_d_vs_pixel=1e-2, eta_perception=1e-2,
                 lambda_attr=0.1, margin_g_d=0.3,
                 generator_optimizer_type=None,  # Original paper used 'rmsprop'
                 discriminator_optimizer_type=None,  # Original paper used 'rmsprop'
                 generator_optimizer_params=None,  # For RMSprop use {'alpha': 0.9}
                 discriminator_optimizer_params=None,  # For RMSprop use {'alpha': 0.9}
                 global_scheduler=None,  # Original paper used 'custom'
                 generator_global_scheduler_params=None,
                 discriminator_global_scheduler_params=None,
                 generator_lambda_lr_decay=0.95,  # LambdaLR decay for the generator
                 discriminator_lambda_lr_decay=0.95,  # LambdaLR decay for the discriminator
                 lambda_d_vs_pixel_decay=0.995,  # Decay for the discriminator weighting
                 fake_attributes='shuffle',  # Either shuffle or invert
                 remove_stn=None,  # By default the model uses STN, if true the STN blocks are removed
                 generator_attribute_encoder=None,
                 discriminator_attribute_encoder=None,
                 # These are a bit of a quick fix to get all the attributes loaded in one place
                 # This is to make sure the fake attributes are the same as the original paper
                 attributes_file_loc=None,
                 attribute_amplification=None,
                 dataset_name='celeba',
                 **kwargs):
        """
        Main handler for Face-Attributes SR GAN.  Please note eval loss is always L1loss, and does not consider extra types of losses.
        """
        super(FaceSRAttributesGANHandler, self).__init__(device=device,
                                                         model_save_dir=model_save_dir,
                                                         eval_mode=eval_mode,
                                                         **kwargs)
        self.net = FaceSRAttributesGeneratorNet(remove_stn=remove_stn, use_attribute_encoder=generator_attribute_encoder)

        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()

        self.lambda_d_vs_pixel = lambda_d_vs_pixel
        self.eta_perception = eta_perception
        self.lambda_attr = lambda_attr
        self.margin_g_d = margin_g_d

        self.lambda_d_vs_pixel_decay = lambda_d_vs_pixel_decay

        self.fake_attributes_func = self.random_shuffle_attributes

        if fake_attributes == 'invert':
            self.fake_attributes_func = self.invert_attributes
        elif fake_attributes == 'choose_from_list':
            self.fake_attributes_func = self.choose_from_list

        self.discriminator_trade_off = False

        if self.margin_g_d > 0:
            self.discriminator_trade_off = True

        self.optimize_discriminator = True
        self.optimize_generator = True

        def lr_lambda_gen(epoch): return max(generator_lambda_lr_decay**epoch, min_gen_lr / generator_lr)
        def lr_lambda_dis(epoch): return max(discriminator_lambda_lr_decay**epoch, min_dis_lr / discriminator_lr)

        if global_scheduler == 'custom':
            generator_global_scheduler_params = {'function': lr_lambda_gen}
            discriminator_global_scheduler_params = {'function': lr_lambda_dis}

        # Specific Optimizer, Discriminator and Scheduler Config (only for training)
        self.optimizer = {}
        self.learning_rate_scheduler = {}

        self.full_dict = {}
        self.dataset_name = dataset_name

        if not self.eval_mode:
            if attributes_file_loc is not None and fake_attributes == 'choose_from_list':
                celeb_data = pd.read_csv(attributes_file_loc, skiprows=1, delim_whitespace=True)

                if attribute_amplification is not None:
                    celeb_data[celeb_data < 0] = -2
                    celeb_data[celeb_data > 0] = 2
                else:
                    celeb_data[celeb_data < 0] = 0

                if self.metadata != 'all':
                    if 'age' in self.metadata:  # certain columns in celeba attributes can have alternate names
                        celeb_data.rename(columns={'Young': 'age'}, inplace=True)
                    if 'gender' in self.metadata:
                        celeb_data.rename(columns={'Male': 'gender'}, inplace=True)
                    celeb_data = celeb_data[self.metadata]

                final_keys = list(celeb_data.columns)
                final_keys.reverse()

                for i in range(len(celeb_data.index)):
                    row_name = celeb_data.index[i]
                    row_list = [celeb_data.loc[row_name][data_key] for data_key in final_keys]
                    self.full_dict[row_name] = row_list

            self.optimizer['generator_optimizer'] = self.define_optimizer(self.net.parameters(),
                                                                          lr=generator_lr,
                                                                          optimizer_params=generator_optimizer_params,
                                                                          optimizer_type=generator_optimizer_type)

            if global_scheduler is not None:
                self.learning_rate_scheduler['generator_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['generator_optimizer'],
                    scheduler=global_scheduler,
                    scheduler_params=generator_global_scheduler_params)

            # Discriminator model
            self.discriminator = FaceSRAttributesDiscriminatorNet(use_attribute_encoder=discriminator_attribute_encoder)
            self.discriminator.to(self.device)
            self.optimizer['discrim_optimizer'] = self.define_optimizer(self.discriminator.parameters(),
                                                                        lr=discriminator_lr,
                                                                        optimizer_params=discriminator_optimizer_params,
                                                                        optimizer_type=discriminator_optimizer_type)

            if global_scheduler is not None:  # same scheduler used for discriminator and for main generator
                self.learning_rate_scheduler['discrim_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['discrim_optimizer'],
                    scheduler=global_scheduler,
                    scheduler_params=discriminator_global_scheduler_params)

            # Perceptual loss mechanism
            self.vgg_extractor = perceptual_loss_mechanism('vgg',
                                                           mode='relu32',
                                                           device=device)
            self.vgg_extractor.to(self.device)
            self.vgg_extractor.eval()

            # Additional error criteria
            self.criterion_discriminator = nn.BCELoss()
            self.criterion_content = nn.MSELoss()

    def run_model(self, x, extra_channels=None, *args, **kwargs):
        return super().run_model(x, extra_channels=extra_channels, *args, **kwargs)

    def run_eval(self, x, y=None, request_loss=False, metadata=None, metadata_keys=None,
                 extra_channels=None, *args, **kwargs):
        input_data, extra_channels = self.channel_concat_logic(x, extra_channels,
                                                               metadata, metadata_keys)

        return super().run_eval(input_data, y, request_loss=request_loss, extra_channels=extra_channels, **kwargs)

    # This is to create adversarial attributes
    # Slightly different than the original paper but should be fine
    # NOTE: Should this be coded differently? Maybe use a random list of numbers instead of shuffling?
    def random_shuffle_attributes(self, attributes):
        idx = torch.randperm(attributes.nelement())
        fake_attributes = attributes.view(-1)[idx].view(attributes.size())
        return fake_attributes

    def invert_attributes(self, attributes):
        fake_attributes = torch.logical_not(attributes).float()
        return fake_attributes

    def choose_from_list(self, attributes, tag_list):
        fake_attributes = torch.clone(attributes)

        num_images = len(self.full_dict.keys())

        if self.dataset_name == 'celeba':
            _, num_images = data_splits['celeba']['train']

        num_attributes = fake_attributes.size(dim=1)
        batch_size = fake_attributes.size(dim=0)

        # List of indices for fake attributes
        random_indices_list = torch.randint(num_images, size=(batch_size*4,))

        last_index = 0

        for i in range(batch_size):
            # Random index of the fake attribute
            chosen_index = 0

            # Go through the generated list of fake attributes
            # Check if the tags are already in the batch
            # Set the random index for an image which is not already in the batch
            # Take the metadata from that image and place it in the tensor
            for j, random_index in enumerate(random_indices_list):
                if j >= last_index:
                    if list(self.full_dict.keys())[random_index.item()] not in tag_list:
                        chosen_index = random_index.item()
                        last_index = j + 1
                        break

            chosen_key = list(self.full_dict.keys())[chosen_index]
            chosen_fake_attributes = torch.reshape(torch.Tensor(self.full_dict[chosen_key]), (1, num_attributes, 1, 1))
            fake_attributes[i, :, :, :] = chosen_fake_attributes

        return fake_attributes

    def discriminator_update(self, ref_image, gen_image, real_attributes, fake_attributes):
        for p in self.discriminator.parameters():
            p.requires_grad = True

        # Three different types of image and attribute combinations:
        # HR Images + Real Attributes (Target = 1)
        # SR Images + Real Attributes (Target = 0)
        # HR Images + Fake Attributes (Target = 0)
        combined_fake_images = torch.cat((gen_image.detach(), ref_image), dim=0)
        combined_fake_attributes = torch.cat((real_attributes, fake_attributes), dim=0)

        combined_fake_images = combined_fake_images.to(device=self.device)
        combined_fake_attributes = combined_fake_attributes.to(device=self.device)

        real_predictions = self.discriminator(ref_image, real_attributes)
        fake_predictions = self.discriminator(combined_fake_images, combined_fake_attributes)

        # Set the target ones
        real_targets = real_predictions.new_ones(real_predictions.size()) * 1.0
        real_targets = real_targets.to(device=self.device)

        # Set the target zeros
        fake_targets = fake_predictions.new_zeros(fake_predictions.size()) * 1.0
        fake_targets = fake_targets.to(device=self.device)

        # Get the loss for the fake and real
        loss_real = self.criterion_discriminator(real_predictions, real_targets)
        loss_fake = self.criterion_discriminator(fake_predictions, fake_targets)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        if self.discriminator_trade_off:
            self.optimize_discriminator = True
            self.optimize_generator = True

            if loss_real < self.margin_g_d or loss_fake < self.margin_g_d:
                self.optimize_discriminator = False

            if loss_real > (1.0 - self.margin_g_d) or loss_fake > (1.0 - self.margin_g_d):
                self.optimize_generator = False

            if not self.optimize_generator and not self.optimize_discriminator:
                self.optimize_generator = True
                self.optimize_discriminator = True

        if self.optimize_discriminator:
            self.optimizer['discrim_optimizer'].zero_grad()
            loss_D.backward()
            self.optimizer['discrim_optimizer'].step()

        return loss_D, loss_real, loss_fake

    def generator_update(self, ref_image, gen_image, real_attributes):
        for p in self.discriminator.parameters():
            p.requires_grad = False

        # MSE Loss
        loss_content = self.criterion_content(gen_image, ref_image)

        # Perceptual Loss (from ReLU3_2 of VGG model)
        gen_vgg_features = self.vgg_extractor(gen_image)
        real_vgg_features = self.vgg_extractor(ref_image.detach())

        loss_perceptual = self.criterion_content(gen_vgg_features, real_vgg_features)  # Divide by 127.5 (like in the original)?

        # Discriminator prediction
        pred_SR = self.discriminator(gen_image, real_attributes)

        target_SR = pred_SR.new_ones(pred_SR.size()) * 1.0
        target_SR = target_SR.to(device=self.device)

        loss_GAN = self.criterion_discriminator(pred_SR, target_SR)

        # This is just to get the gradients for the loss function
        # NOTE: Is this needed?
        # loss_GAN.backward()

        # TODO: FINISH THE GRAD INPUT HOOK
        # MAYBE CHECK THIS:
        # https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/#:~:text=PyTorch%20provides%20two%20types%20of%20hooks.&text=A%20forward%20hook%20is%20executed,Function%20object.
        # NOTE: Not exactly sure about this
        # Original loss: https://github.com/XinYuANU/FaceAttr/blob/0650116bd1e374eb16ad8149f691a0a0b35fabdb/adversarial_xin_Attr_AE_Stack_perception.lua#L271
        # grad_in = self.discriminator.discriminator_first_layer.weight.grad
        # loss_G = grad_in * self.lambda_d_vs_pixel + loss_content + loss_perceptual * self.eta_perception

        loss_G = loss_GAN * self.lambda_d_vs_pixel + loss_content + loss_perceptual * self.eta_perception

        if self.optimize_generator:
            self.optimizer['generator_optimizer'].zero_grad()
            loss_G.backward()
            self.optimizer['generator_optimizer'].step()

        return loss_G, loss_content, loss_perceptual, loss_GAN

    def run_train(self, x, y, metadata, extra_channels=None, metadata_keys=None, *args, **kwargs):
        """
        Runs one training iteration (pre-train or GAN) through a data batch
        """
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')

        # sets model to training mode (activates appropriate procedures for certain layers)
        self.net.train()
        self.discriminator.train()

        # Get the metadata for the model
        input_data, real_attributes = self.channel_concat_logic(x, extra_channels, metadata, metadata_keys)

        input_data = input_data.to(device=self.device)
        y = y.to(device=self.device)

        real_attributes = real_attributes.to(device=self.device)

        if self.fake_attributes_func.__name__ == 'choose_from_list':
            fake_attributes = self.fake_attributes_func(real_attributes, kwargs['tag']).to(device=self.device)
        else:
            fake_attributes = self.fake_attributes_func(real_attributes).to(device=self.device)

        out = self.run_model(input_data, real_attributes)

        # Decisions on whether to optimise the discriminator and generator are handled directly
        # in their respective update functions
        loss_D, loss_real, loss_fake = self.discriminator_update(y, out, real_attributes, fake_attributes)
        loss_G, loss_content, loss_perceptual, loss_GAN = self.generator_update(y, out, real_attributes)

        loss_package = {}

        for _loss, name in zip((loss_G, loss_content, loss_GAN, loss_perceptual, loss_D, loss_real, loss_fake),
                               ('train-loss', 'l2-loss', 'gan-loss', 'vgg-loss', 'discriminator-loss', 'd-loss-real', 'd-loss-fake')):
            loss_package[name] = _loss.cpu().data.numpy()

        return loss_package, out.detach().cpu()

    def epoch_end_calls(self):  # only call learning rate scheduler once per epoch
        if 'generator_scheduler' in self.learning_rate_scheduler:
            self.learning_rate_scheduler['generator_scheduler'].step()

        if 'discrim_scheduler' in self.learning_rate_scheduler:
            self.learning_rate_scheduler['discrim_scheduler'].step()

        self.lambda_d_vs_pixel = max(self.lambda_d_vs_pixel * self.lambda_d_vs_pixel_decay, 0.005)

    def get_learning_rate(self):  # TODO: this could also be generalised for all multi discriminator models
        lrs = {}
        for key, optimizer in self.optimizer.items():
            lrs['%s_learning_rate' % key] = optimizer.param_groups[0]['lr']
        return lrs

    def extra_diagnostics(self):
        if not self.eval_mode:
            models = [self.net, self.discriminator, self.vgg_extractor]
            model_names = ['Generator', 'Discriminator', 'VGG Extractor']
            self.print_parameters_model_list(models, model_names)


class AGAGANHandler(QModel):
    def __init__(self,
                 device, model_save_dir, eval_mode=False, scale=8,
                 generator_lr=1e-4,
                 discriminator_lr=1e-4,
                 unet_lr=1e-4,
                 lambda_pixel=0.75,
                 lambda_perceptual=0.25,
                 lambda_discriminator=0.003,
                 n_attributes=40,  # Paper uses 38 but we're unsure which, we use 40 for now
                 pre_unet_epochs=None,
                 generator_optimizer_type='adam',
                 discriminator_optimizer_type='adam',
                 unet_optimizer_type='adam',
                 generator_optimizer_params=None,  # None for default Adam params
                 discriminator_optimizer_params=None,  # None for default Adam params
                 unet_optimizer_params=None,  # None for default Adam params
                 global_scheduler=None,
                 generator_global_scheduler_params=None,  # Not used but keeping them for flexibility
                 discriminator_global_scheduler_params=None,
                 unet_global_scheduler_params=None,
                 use_scheduler=False,
                 **kwargs):
        super(AGAGANHandler, self).__init__(device=device,
                                            model_save_dir=model_save_dir,
                                            eval_mode=eval_mode,
                                            **kwargs)

        # Paper states to use 38 attributes, but at the moment we're not full sure which where removed
        # For now, we will go with all the 40 attributes
        self.net = AGAGANGenerator(n_attributes=n_attributes)
        self.unet = AGAGANUNet()
        self.unet.to(device=self.device)

        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()

        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_discriminator = lambda_discriminator

        self.scale = scale

        self.pre_unet_epochs = pre_unet_epochs
        self.use_scheduler = use_scheduler

        # Specific Optimizer, Discriminator and Scheduler Config (only for training)
        self.optimizer = {}
        self.learning_rate_scheduler = {}

        if not self.eval_mode:
            self.optimizer['generator_optimizer'] = self.define_optimizer(self.net.parameters(),
                                                                          lr=generator_lr,
                                                                          optimizer_params=generator_optimizer_params,
                                                                          optimizer_type=generator_optimizer_type)

            if global_scheduler is not None:
                self.learning_rate_scheduler['generator_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['generator_optimizer'],
                    scheduler=global_scheduler,
                    scheduler_params=generator_global_scheduler_params)

            # Discriminator model
            # Similar to the generator, the paper states to use 38 attributes
            # But at the moment we're not full sure which where removed
            # For now, we will go with all the 40 attributes
            self.discriminator = AGAGANDiscriminatorNet(n_attributes=n_attributes)
            self.discriminator.to(self.device)
            self.optimizer['discrim_optimizer'] = self.define_optimizer(self.discriminator.parameters(),
                                                                        lr=discriminator_lr,
                                                                        optimizer_params=discriminator_optimizer_params,
                                                                        optimizer_type=discriminator_optimizer_type)

            if global_scheduler is not None:  # same scheduler used for discriminator and for main generator
                self.learning_rate_scheduler['discrim_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['discrim_optimizer'],
                    scheduler=global_scheduler,
                    scheduler_params=discriminator_global_scheduler_params)

            self.optimizer['unet_optimizer'] = self.define_optimizer(self.unet.parameters(),
                                                                     lr=unet_lr,
                                                                     optimizer_params=unet_optimizer_params,
                                                                     optimizer_type=unet_optimizer_type)

            if global_scheduler is not None:  # same scheduler used for discriminator and for main generator
                self.learning_rate_scheduler['unet_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['unet_optimizer'],
                    scheduler=global_scheduler,
                    scheduler_params=unet_global_scheduler_params)

            # Perceptual loss mechanism
            self.vgg_extractor = perceptual_loss_mechanism('vgg',
                                                           mode='conv54',
                                                           device=device)
            self.vgg_extractor.to(self.device)
            self.vgg_extractor.eval()

            # Additional error criteria
            self.criterion_discriminator = nn.BCELoss()
            self.criterion_MSE = nn.MSELoss()
            self.criterion_MAE = nn.L1Loss()

    def run_model(self, x, extra_channels=None, *args, **kwargs):
        out = self.net.forward(x, metadata=extra_channels)

        if self.curr_epoch < self.pre_unet_epochs:
            return out

        x_up_bicubic = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x_up_bicubic = x_up_bicubic.to(device=self.device)
        unet_input = torch.cat((out, x_up_bicubic), dim=1)

        unet_out = self.unet(unet_input)

        return unet_out

    def run_eval(self, x, y=None, request_loss=False, metadata=None, metadata_keys=None,
                 extra_channels=None, *args, **kwargs):
        input_data, extra_channels = self.channel_concat_logic(x, extra_channels,
                                                               metadata, metadata_keys)

        return super().run_eval(input_data, y, request_loss=request_loss, extra_channels=extra_channels, **kwargs)

    def discriminator_update(self, ref_image, gen_image, attributes):
        for p in self.discriminator.parameters():
            p.requires_grad = True

        # Real and fake predictions
        real_predictions = self.discriminator(ref_image, attributes)
        fake_predictions = self.discriminator(gen_image.detach(), attributes)

        # Real and fake targets
        # Not fully sure why the fake targets are random, but it's an interesting idea
        real_targets = (real_predictions.new_ones(real_predictions.size()) * 1.0) - \
            (torch.rand(real_predictions.size(), device=self.device) * 0.2)
        fake_targets = torch.rand(fake_predictions.size(), device=self.device) * 0.2

        loss_real = self.criterion_discriminator(real_predictions, real_targets)
        loss_fake = self.criterion_discriminator(fake_predictions, fake_targets)

        loss_D = (loss_real + loss_fake) / 2

        self.optimizer['discrim_optimizer'].zero_grad()
        loss_D.backward()
        self.optimizer['discrim_optimizer'].step()

        return loss_D, loss_real, loss_fake

    def generator_update(self, ref_image, gen_image, attributes):
        for p in self.discriminator.parameters():
            p.requires_grad = False

        # Pixel loss
        loss_content = self.criterion_MAE(gen_image, ref_image)

        # Perceptual loss
        gen_vgg_features = self.vgg_extractor(gen_image)
        real_vgg_features = self.vgg_extractor(ref_image.detach())

        loss_perceptual = self.criterion_MSE(gen_vgg_features, real_vgg_features)

        # Discriminator loss
        pred_SR = self.discriminator(gen_image, attributes)

        target_SR = (pred_SR.new_ones(pred_SR.size()) * 1.0) - (torch.rand(pred_SR.size(), device=self.device) * 0.2)
        target_SR = target_SR.to(device=self.device)

        loss_GAN = self.criterion_discriminator(pred_SR, target_SR)

        # Combined loss: 0.75*pixel + 0.25*perceptual + 0.003*discriminator
        loss_G = (self.lambda_pixel * loss_content) + (self.lambda_perceptual * loss_perceptual) + (self.lambda_discriminator * loss_GAN)

        self.optimizer['generator_optimizer'].zero_grad()
        loss_G.backward()
        self.optimizer['generator_optimizer'].step()

        return loss_G, loss_content, loss_perceptual, loss_GAN

    def unet_update(self, ref_image, unet_image):
        for p in self.net.parameters():
            p.requires_grad = False

        for p in self.discriminator.parameters():
            p.requires_grad = False

        loss_content = self.criterion_MAE(unet_image, ref_image)

        gen_vgg_features = self.vgg_extractor(unet_image)
        real_vgg_features = self.vgg_extractor(ref_image.detach())

        loss_perceptual = self.criterion_MSE(gen_vgg_features, real_vgg_features)

        loss_unet = (self.lambda_pixel * loss_content) + (self.lambda_perceptual * loss_perceptual)

        self.optimizer['unet_optimizer'].zero_grad()
        loss_unet.backward()
        self.optimizer['unet_optimizer'].step()

        return loss_unet

    def run_train(self, x, y, metadata, extra_channels=None, metadata_keys=None, *args, **kwargs):
        """
        Runs one training iteration (pre-train or GAN) through a data batch
        """
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')

        # sets model to training mode (activates appropriate procedures for certain layers)
        self.net.train()
        self.discriminator.train()
        self.unet.train()

        # Get the metadata for the model
        input_data, attributes = self.channel_concat_logic(x, extra_channels, metadata, metadata_keys)

        input_data = input_data.to(device=self.device)
        attributes = attributes.to(device=self.device)

        y = y.to(device=self.device)

        # out = self.run_model(input_data, attributes)
        out = self.net.forward(input_data, metadata=attributes)  # Use forward directly, since self.run_model includes the UNet

        if self.curr_epoch < self.pre_unet_epochs:
            loss_D, loss_real, loss_fake = self.discriminator_update(y, out, attributes)
            loss_G, loss_content, loss_perceptual, loss_GAN = self.generator_update(y, out, attributes)
            loss_unet = torch.tensor(0.0, device=self.device)
        else:
            lr_image_bicubic = F.interpolate(input_data, scale_factor=self.scale, mode='bicubic', align_corners=False)
            lr_image_bicubic = lr_image_bicubic.to(device=self.device)

            unet_input = torch.cat((out, lr_image_bicubic), dim=1)

            unet_out = self.unet(unet_input)

            loss_unet = self.unet_update(y, unet_out)

            loss_D = torch.tensor(0.0, device=self.device)
            loss_real = torch.tensor(0.0, device=self.device)
            loss_fake = torch.tensor(0.0, device=self.device)
            loss_G = torch.tensor(0.0, device=self.device)
            loss_content = torch.tensor(0.0, device=self.device)
            loss_perceptual = torch.tensor(0.0, device=self.device)
            loss_GAN = torch.tensor(0.0, device=self.device)

        loss_package = {}

        for _loss, name in zip((loss_G, loss_content, loss_GAN, loss_perceptual, loss_D, loss_real, loss_fake, loss_unet),
                               ('train-loss', 'l1-loss', 'gan-loss', 'vgg-loss', 'discriminator-loss', 'd-loss-real', 'd-loss-fake', 'unet-loss')):
            loss_package[name] = _loss.cpu().data.numpy()

        return loss_package, out.detach().cpu()

    def epoch_end_calls(self):  # only call learning rate scheduler once per epoch
        if self.use_scheduler:
            if self.curr_epoch < self.pre_unet_epochs:
                if 'generator_scheduler' in self.learning_rate_scheduler:
                    self.learning_rate_scheduler['generator_scheduler'].step()

                if 'discrim_scheduler' in self.learning_rate_scheduler:
                    self.learning_rate_scheduler['discrim_scheduler'].step()
            else:
                if 'unet_scheduler' in self.learning_rate_scheduler:
                    self.learning_rate_scheduler['unet_scheduler'].step()

    def get_learning_rate(self):  # TODO: this could also be generalised for all multi discriminator models
        lrs = {}
        for key, optimizer in self.optimizer.items():
            lrs['%s_learning_rate' % key] = optimizer.param_groups[0]['lr']
        return lrs

    def extra_diagnostics(self):
        if not self.eval_mode:
            models = [self.net, self.unet, self.discriminator, self.vgg_extractor]
            model_names = ['Generator', 'UNet', 'Discriminator', 'VGG Extractor']
            self.print_parameters_model_list(models, model_names)

    def save_model(self, model_save_name, extract_state_only=True, minimal=False):
        super().save_model(model_save_name=model_save_name,
                           extract_state_only=extract_state_only,
                           minimal=minimal)

        if hasattr(self, 'unet'):
            self._extract_multiple_models_from_dict(self.unet, 'unet', config='save')

        torch.save(self.state, f=os.path.join(self.model_save_dir, "{}_{}".format(model_save_name, self.curr_epoch)))

    def load_model(self, model_save_name, model_idx, legacy=False, load_override=None, preloaded_state=None):
        state = super().load_model(model_save_name=model_save_name,
                                   model_idx=model_idx,
                                   legacy=legacy,
                                   load_override=load_override,
                                   preloaded_state=preloaded_state)

        if hasattr(self, 'unet'):
            self._extract_multiple_models_from_dict(self.unet, 'unet', config='load', load_state=state)

        return state


class FMFNetHandler(QModel):
    def __init__(self,
                 device, model_save_dir, eval_mode=False, scale=8,
                 generator_lr=1e-4,
                 discriminator_lr=1e-4,
                 attribute_discriminator_lr=1e-4,
                 lambda_pixel=1.0,
                 lambda_perceptual=2e-1,
                 lambda_discriminator=1e-1,
                 lambda_attribute_discriminator=2e-1,
                 vgg_layer_output='relu32',
                 n_attributes=40,
                 generator_optimizer_type='adam',
                 discriminator_optimizer_type='adam',
                 attribute_discriminator_optimizer_type='adam',
                 generator_optimizer_params=None,  # None for default Adam params
                 discriminator_optimizer_params=None,  # None for default Adam params
                 attribute_discriminator_optimizer_params=None,  # None for default Adam params
                 global_scheduler=None,
                 generator_global_scheduler_params=None,  # Not used but keeping them for flexibility
                 discriminator_global_scheduler_params=None,
                 attribute_discriminator_global_scheduler_params=None,
                 use_esrgan_discriminator=False,
                 use_esrgan_style_discriminator_loss=False,
                 use_sigmoid_discriminator=True,
                 use_sigmoid_attribute_discriminator=True,
                 use_scheduler=True,
                 update_scheduler_per_batch=False,
                 use_meta_attention=True,
                 latent_dim_size_factor=1.0,
                 **kwargs):
        super(FMFNetHandler, self).__init__(device=device,
                                            model_save_dir=model_save_dir,
                                            eval_mode=eval_mode,
                                            **kwargs)

        self.net = FMFResidualDenseNet(n_attributes=n_attributes, use_meta_attention=use_meta_attention, latent_dim_size_factor=latent_dim_size_factor)

        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()

        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_discriminator = lambda_discriminator
        self.lambda_attribute_discriminator = lambda_attribute_discriminator

        self.scale = scale

        self.use_esrgan_style_discriminator_loss = use_esrgan_style_discriminator_loss

        self.use_sigmoid_discriminator = use_sigmoid_discriminator
        self.use_sigmoid_attribute_discriminator = use_sigmoid_attribute_discriminator

        self.use_scheduler = use_scheduler
        self.update_scheduler_per_batch = update_scheduler_per_batch

        # Specific Optimizer, Discriminator and Scheduler Config (only for training)
        self.optimizer = {}
        self.learning_rate_scheduler = {}

        if not self.eval_mode:
            self.optimizer['generator_optimizer'] = self.define_optimizer(self.net.parameters(),
                                                                          lr=generator_lr,
                                                                          optimizer_params=generator_optimizer_params,
                                                                          optimizer_type=generator_optimizer_type)

            if global_scheduler is not None:
                self.learning_rate_scheduler['generator_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['generator_optimizer'],
                    scheduler=global_scheduler,
                    scheduler_params=generator_global_scheduler_params)

            # Image Discriminator Model
            self.discriminator = FMFDiscriminator(use_sigmoid=use_sigmoid_discriminator)

            if use_esrgan_discriminator:
                self.discriminator = VGGStyleDiscriminator128()

            self.discriminator.to(self.device)
            self.optimizer['discrim_optimizer'] = self.define_optimizer(self.discriminator.parameters(),
                                                                        lr=discriminator_lr,
                                                                        optimizer_params=discriminator_optimizer_params,
                                                                        optimizer_type=discriminator_optimizer_type)

            if global_scheduler is not None:  # same scheduler used for discriminator and for main generator
                self.learning_rate_scheduler['discrim_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['discrim_optimizer'],
                    scheduler=global_scheduler,
                    scheduler_params=discriminator_global_scheduler_params)

            # Attribute Discriminator Model
            # The idea for this model is to ensure that generated images contain the expected features
            self.attribute_discriminator = FMFAttributeDiscriminator(n_attributes=n_attributes, use_sigmoid=use_sigmoid_attribute_discriminator)
            self.attribute_discriminator.to(self.device)
            self.optimizer['attr_discrim_optimizer'] = self.define_optimizer(self.attribute_discriminator.parameters(),
                                                                             lr=attribute_discriminator_lr,
                                                                             optimizer_params=attribute_discriminator_optimizer_params,
                                                                             optimizer_type=attribute_discriminator_optimizer_type)

            if global_scheduler is not None:  # same scheduler used for discriminator and for main generator
                self.learning_rate_scheduler['attr_discrim_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['attr_discrim_optimizer'],
                    scheduler=global_scheduler,
                    scheduler_params=attribute_discriminator_global_scheduler_params)

            # Perceptual loss mechanism
            self.vgg_extractor = perceptual_loss_mechanism('vgg',
                                                           mode=vgg_layer_output,
                                                           device=device)
            self.vgg_extractor.to(self.device)
            self.vgg_extractor.eval()

            # Additional error criteria
            self.criterion_discriminator = nn.BCELoss()
            if use_esrgan_discriminator or not use_sigmoid_discriminator or use_esrgan_style_discriminator_loss:
                self.criterion_discriminator = nn.BCEWithLogitsLoss()

            self.criterion_attribute_discriminator = nn.BCELoss()
            if not use_sigmoid_attribute_discriminator:
                self.criterion_attribute_discriminator = nn.BCEWithLogitsLoss()

            self.criterion_MSE = nn.MSELoss()
            self.criterion_MAE = nn.L1Loss()

    def run_model(self, x, extra_channels=None, *args, **kwargs):
        return super().run_model(x, extra_channels=extra_channels, *args, **kwargs)

    def run_eval(self, x, y=None, request_loss=False, metadata=None, metadata_keys=None,
                 extra_channels=None, *args, **kwargs):
        input_data, extra_channels = self.channel_concat_logic(x, extra_channels,
                                                               metadata, metadata_keys)

        return super().run_eval(input_data, y, request_loss=request_loss, extra_channels=extra_channels, **kwargs)

    def discriminator_update(self, ref_image, gen_image):
        for p in self.discriminator.parameters():
            p.requires_grad = True

        # Real and fake predictions
        real_predictions = self.discriminator(ref_image)
        fake_predictions = self.discriminator(gen_image.detach())

        # Real and fake targets
        real_targets = real_predictions.new_ones(real_predictions.size()) * 1.0
        fake_targets = fake_predictions.new_ones(fake_predictions.size()) * 0.0

        if self.use_esrgan_style_discriminator_loss and not self.use_sigmoid_discriminator:
            loss_real = self.criterion_discriminator(real_predictions - fake_predictions.mean(0, keepdim=True), real_targets)
            loss_fake = self.criterion_discriminator(fake_predictions - real_predictions.mean(0, keepdim=True), fake_targets)
        else:
            loss_real = self.criterion_discriminator(real_predictions, real_targets)
            loss_fake = self.criterion_discriminator(fake_predictions, fake_targets)

        loss_D = (loss_real + loss_fake) / 2

        self.optimizer['discrim_optimizer'].zero_grad()
        loss_D.backward()
        self.optimizer['discrim_optimizer'].step()

        if self.use_scheduler:
            if self.update_scheduler_per_batch:
                if 'discrim_scheduler' in self.learning_rate_scheduler:
                    self.learning_rate_scheduler['discrim_scheduler'].step()

        return loss_D, loss_real, loss_fake

    def attribute_discriminator_update(self, ref_image, attributes):
        for p in self.attribute_discriminator.parameters():
            p.requires_grad = True

        # Real predictions
        real_predictions = self.attribute_discriminator(ref_image.detach())

        # Real target attributes
        real_target_attributes = torch.squeeze(attributes)
        real_target_attributes = real_target_attributes.to(device=self.device)

        loss_attribute_D = self.criterion_attribute_discriminator(real_predictions, real_target_attributes)

        self.optimizer['attr_discrim_optimizer'].zero_grad()
        loss_attribute_D.backward()
        self.optimizer['attr_discrim_optimizer'].step()

        if self.use_scheduler:
            if self.update_scheduler_per_batch:
                if 'attr_discrim_scheduler' in self.learning_rate_scheduler:
                    self.learning_rate_scheduler['attr_discrim_scheduler'].step()

        return loss_attribute_D

    def generator_update(self, ref_image, gen_image, attributes):
        for p in self.discriminator.parameters():
            p.requires_grad = False

        for p in self.attribute_discriminator.parameters():
            p.requires_grad = False

         # Pixel loss
        loss_content = self.criterion_MAE(gen_image, ref_image)

        # Perceptual loss
        gen_vgg_features = self.vgg_extractor(gen_image)
        real_vgg_features = self.vgg_extractor(ref_image.detach())

        loss_perceptual = self.criterion_MSE(gen_vgg_features, real_vgg_features)

        # Discriminator loss
        real_predictions = self.discriminator(ref_image.detach())
        SR_predictions = self.discriminator(gen_image)

        real_targets = real_predictions.new_ones(real_predictions.size()) * 1.0
        real_targets = real_targets.to(device=self.device)

        SR_targets = SR_predictions.new_ones(SR_predictions.size()) * 0.0
        SR_targets = SR_targets.to(device=self.device)

        if self.use_esrgan_style_discriminator_loss and not self.use_sigmoid_discriminator:
            loss_real = self.criterion_discriminator(real_predictions - SR_predictions.mean(0, keepdim=True), SR_targets)
            loss_fake = self.criterion_discriminator(SR_predictions - real_predictions.mean(0, keepdim=True), real_targets)

            loss_GAN = (loss_real + loss_fake) / 2
        else:
            loss_GAN = self.criterion_discriminator(SR_predictions, SR_targets)

        # Attribute discriminator loss
        pred_attr = self.attribute_discriminator(gen_image)

        target_attr = torch.squeeze(attributes)
        target_attr = target_attr.to(device=self.device)

        loss_AD_GAN = self.criterion_attribute_discriminator(pred_attr, target_attr)

        # Combined loss
        loss_G = (self.lambda_pixel * loss_content) + (self.lambda_perceptual * loss_perceptual) +\
                 (self.lambda_discriminator * loss_GAN) + (self.lambda_attribute_discriminator * loss_AD_GAN)

        self.optimizer['generator_optimizer'].zero_grad()
        loss_G.backward()
        self.optimizer['generator_optimizer'].step()

        if self.use_scheduler:
            if self.update_scheduler_per_batch:
                if 'generator_scheduler' in self.learning_rate_scheduler:
                    self.learning_rate_scheduler['generator_scheduler'].step()

        return loss_G, loss_content, loss_perceptual, loss_GAN, loss_AD_GAN

    def run_train(self, x, y, metadata, extra_channels=None, metadata_keys=None, *args, **kwargs):
        """
        Runs one training iteration (pre-train or GAN) through a data batch
        """
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')

        # sets model to training mode (activates appropriate procedures for certain layers)
        self.net.train()
        self.discriminator.train()
        self.attribute_discriminator.train()

        # Get the metadata for the model
        input_data, attributes = self.channel_concat_logic(x, extra_channels, metadata, metadata_keys)

        input_data = input_data.to(device=self.device)
        attributes = attributes.to(device=self.device)

        y = y.to(device=self.device)

        out = self.run_model(input_data, attributes)

        loss_D, loss_real, loss_fake = self.discriminator_update(y, out)
        loss_attribute_D = self.attribute_discriminator_update(y, attributes)
        loss_G, loss_content, loss_perceptual, loss_GAN, loss_AD_GAN = self.generator_update(y, out, attributes)

        loss_package = {}

        for _loss, name in zip((loss_G, loss_content, loss_GAN, loss_AD_GAN, loss_perceptual, loss_D, loss_real, loss_fake, loss_attribute_D),
                               ('train-loss', 'l1-loss', 'gan-loss', 'attribute-gan-loss', 'vgg-loss', 'discriminator-loss', 'd-loss-real', 'd-loss-fake', 'attribute-discriminator-loss')):
            loss_package[name] = _loss.cpu().data.numpy()

        return loss_package, out.detach().cpu()

    def epoch_end_calls(self):  # only call learning rate scheduler once per epoch
        if self.use_scheduler:
            if not self.update_scheduler_per_batch:
                if 'generator_scheduler' in self.learning_rate_scheduler:
                    self.learning_rate_scheduler['generator_scheduler'].step()

                if 'discrim_scheduler' in self.learning_rate_scheduler:
                    self.learning_rate_scheduler['discrim_scheduler'].step()

                if 'attr_discrim_scheduler' in self.learning_rate_scheduler:
                    self.learning_rate_scheduler['attr_discrim_scheduler'].step()

    def get_learning_rate(self):  # TODO: this could also be generalised for all multi discriminator models
        lrs = {}
        for key, optimizer in self.optimizer.items():
            lrs['%s_learning_rate' % key] = optimizer.param_groups[0]['lr']
        return lrs

    def extra_diagnostics(self):
        if not self.eval_mode:
            models = [self.net, self.discriminator, self.attribute_discriminator, self.vgg_extractor]
            model_names = ['Generator', 'Discriminator', 'Attribute Discriminator', 'VGG Extractor']
            self.print_parameters_model_list(models, model_names)

    def save_model(self, model_save_name, extract_state_only=True, minimal=False):
        super().save_model(model_save_name=model_save_name,
                           extract_state_only=extract_state_only,
                           minimal=minimal)

        if hasattr(self, 'attribute_discriminator'):
            self._extract_multiple_models_from_dict(self.attribute_discriminator, 'attribute_discriminator', config='save')

        torch.save(self.state, f=os.path.join(self.model_save_dir, "{}_{}".format(model_save_name, self.curr_epoch)))

    def load_model(self, model_save_name, model_idx, legacy=False, load_override=None, preloaded_state=None):
        state = super().load_model(model_save_name=model_save_name,
                                   model_idx=model_idx,
                                   legacy=legacy,
                                   load_override=load_override,
                                   preloaded_state=preloaded_state)

        if hasattr(self, 'attribute_discriminator'):
            self._extract_multiple_models_from_dict(self.attribute_discriminator,
                                                    'attribute_discriminator', config='load', load_state=state)

        return state
