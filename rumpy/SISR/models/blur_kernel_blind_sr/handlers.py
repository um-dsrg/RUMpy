import torch
from torch import nn
import numpy as np
import time
import os
import toml
import csv

from rumpy.SISR.models.non_blind_gan_models import BaseBSRGANModel
from rumpy.SISR.models.non_blind_gan_models.discriminators import *
from rumpy.SISR.models.feature_extractors.handlers import perceptual_loss_mechanism
from rumpy.SISR.models.blur_kernel_blind_sr.DASR import DASRPipeline
from rumpy.SISR.models.blur_kernel_blind_sr.contrastive_blind_sr import ContrastiveBlindSRPipeline
from rumpy.SISR.models.blur_kernel_blind_sr.DANv1 import DAN
from rumpy.SISR.models.blur_kernel_blind_sr.DANv2 import DANv2
from rumpy.SISR.models.blur_kernel_blind_sr.DANv1Models import DANv1QRCAN, DANv1QHAN, DANv1QELAN, DANv1QRRDB
from rumpy.SISR.models.blur_kernel_blind_sr.IKC import Predictor, Corrector
from rumpy.regression.models.contrastive_learning import BaseContrastive
from rumpy.shared_framework.models.base_architecture import BaseModel, MultiModel
from rumpy.sr_tools.stats import load_statistics
from rumpy.sr_tools.helper_functions import standard_metric_epoch_selection
from rumpy.SISR.models.attention_manipulators.mini_model import Metabed
from rumpy.SISR.models.attention_manipulators.architectures import QRCAN, QHAN, QEDSR, QELAN, QSAN
from rumpy.SISR.models.attention_manipulators.handlers import QRealESRGANHandler


class DANHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, perceptual=None, mode='v1',
                 optimizer_params=None, scheduler=None, scheduler_params=None,
                 use_pca_encoder=True, selected_metadata=None, pre_trained_estimator_weights=None, **kwargs):
        super(DANHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                         **kwargs)
        if mode == 'v1':
            self.net = DAN(upscale=scale, **kwargs)
        elif mode == 'v2':
            self.net = DANv2(upscale=scale, **kwargs)
        elif 'v1Q' in mode:
            input_para = 10
            if selected_metadata:
                input_para = len(selected_metadata)

            if mode == 'v1QRCAN':
                self.net = DANv1QRCAN(upscale=scale, input_para=input_para, use_pca_encoder=use_pca_encoder, **kwargs)
            elif mode == 'v1QHAN':
                self.net = DANv1QHAN(upscale=scale, input_para=input_para, use_pca_encoder=use_pca_encoder, **kwargs)
            elif mode == 'v1QELAN':
                self.net = DANv1QELAN(upscale=scale, input_para=input_para, use_pca_encoder=use_pca_encoder, **kwargs)
        else:
            raise NotImplementedError('Set mode to v1 or v2')

        if pre_trained_estimator_weights and not kwargs['checkpoint_load']:
            if 'v1' in mode:
                if device == torch.device('cpu'):
                    f_device = device
                else:
                    f_device = "cuda:%d" % device

                state = torch.load(f=pre_trained_estimator_weights, map_location=f_device)

                estimator_dict = {}
                for key, val in state['network'].items():
                    if 'Estimator' in key:
                        estimator_dict[key[10:]] = val

                self.net.Estimator.load_state_dict(state_dict=estimator_dict)

        self.mode = mode
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device, optimizer_params=optimizer_params)
        self.model_name = 'dan'
        self.selected_metadata = selected_metadata
        if scheduler == 'multi_step_lr':
            self.end_epoch_scheduler = True
        else:
            self.end_epoch_scheduler = False

    def run_train(self, x, y, metadata=None, metadata_keys=None, blur_kernels=None, *args, **kwargs):
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)
        x, y, metadata = x.to(device=self.device), y.to(device=self.device), metadata.to(device=self.device)
        if self.mode == 'v2':
            if blur_kernels is None:
                raise RuntimeError('Full blur kernels required for training DANv2.')
            blur_kernels = blur_kernels.to(device=self.device)
            # TODO: be careful here if metadata is not just blur kernels....

        if self.mode == 'v2':
            srs, kernel_maps, kernels = self.run_model(x, train=True)
        else:
            srs, kernel_maps = self.run_model(x, train=True)

        loss_package = {}
        d_sr = 0
        d_kr = 0
        for ind in range(len(kernel_maps)):
            if self.mode == 'v2':
                # V2 model uses entire blur kernel for loss
                d_kr = self.criterion(kernels[ind], blur_kernels.view(*kernels[ind].shape))
            else:
                # V1 model uses PCA-ed blur kernel for loss
                if self.selected_metadata:
                    key_index = [i for i, keys in enumerate(metadata_keys) if keys[0] in self.selected_metadata]
                    d_kr = self.criterion(kernel_maps[ind], metadata[:, key_index])
                else:
                    d_kr = self.criterion(kernel_maps[ind], metadata)

            d_sr = self.criterion(srs[ind], y)
            loss_package["image-loss-iter-%d" % ind] = d_sr.cpu().data.numpy()
            loss_package["kernel-loss-iter-%d" % ind] = d_kr.cpu().data.numpy()

        final_loss = d_sr + d_kr

        self.standard_update(final_loss, scheduler_skip=self.end_epoch_scheduler)

        loss_package['train-loss'] = final_loss.cpu().data.numpy()
        return loss_package, srs[-1].detach().cpu()

    def run_model(self, x, train=False, *args, **kwargs):

        all_data = self.net.forward(x)

        if train:
            return all_data
        else:
            norm_path = os.path.normpath(self.model_save_dir)
            split_path = norm_path.split(os.sep)

            return all_data[0][-1]

    def epoch_end_calls(self):
        if self.learning_rate_scheduler is not None and isinstance(self.learning_rate_scheduler,
                                                                   torch.optim.lr_scheduler.MultiStepLR):
            self.learning_rate_scheduler.step()


class DANv1QRealESRGANHandler(BaseBSRGANModel):
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4,
                 use_pca_encoder=True,
                 selected_metadata=None,
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
        super(DANv1QRealESRGANHandler, self).__init__(device=device,
                                                      model_save_dir=model_save_dir,
                                                      eval_mode=eval_mode,
                                                      **kwargs)

        input_para = 10
        if selected_metadata:
            input_para = len(selected_metadata)

        self.net = DANv1QRRDB(upscale=scale, input_para=input_para, use_pca_encoder=use_pca_encoder, **kwargs)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.model_name = 'danv1qrealesrgan'
        self.selected_metadata = selected_metadata

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

    def generator_update(self, gen_image, ref_image, dan_loss):
        for p in self.discriminator.parameters():  # TODO: is this really needed?
            p.requires_grad = False

        # L1 loss
        loss = dan_loss

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

    def run_train(self, x, y, metadata=None, metadata_keys=None, blur_kernels=None, *args, **kwargs):
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)
        self.discriminator.train()

        x, y, metadata = x.to(device=self.device), y.to(device=self.device), metadata.to(device=self.device)
        srs, kernel_maps = self.run_model(x, train=True)

        loss_package = {}
        d_sr = 0
        d_kr = 0
        for ind in range(len(kernel_maps)):
            if self.selected_metadata:
                key_index = [i for i, keys in enumerate(metadata_keys) if keys[0] in self.selected_metadata]
                d_kr = self.criterion(kernel_maps[ind], metadata[:, key_index])
            else:
                d_kr = self.criterion(kernel_maps[ind], metadata)

            d_sr = self.criterion(srs[ind], y)
            loss_package["image-loss-iter-%d" % ind] = d_sr.cpu().data.numpy()
            loss_package["kernel-loss-iter-%d" % ind] = d_kr.cpu().data.numpy()

        dan_loss = d_sr + d_kr

        if self.curr_epoch < self.pretrain_epochs:  # L1 pre-training
            self.pre_train_update(dan_loss)

            loss = dan_loss
            loss_G = loss
            loss_content = torch.tensor(0.0, device=self.device)
            loss_GAN = torch.tensor(0.0, device=self.device)
            loss_real = torch.tensor(0.0, device=self.device)
            loss_fake = torch.tensor(0.0, device=self.device)
        else:
            loss_G, loss_content, loss_GAN, loss = self.generator_update(srs[-1], y, dan_loss)
            loss_real, loss_fake = self.discriminator_update(srs[-1], y)

        for _loss, name in zip((loss_G, loss, loss_GAN, loss_content, loss_real, loss_fake),
                               ('train-loss', 'l1-loss', 'gan-loss', 'vgg-loss', 'd-loss-real', 'd-loss-fake')):
            loss_package[name] = _loss.cpu().data.numpy()

        return loss_package, srs[-1].detach().cpu()

    def run_model(self, x, train=False, *args, **kwargs):

        all_data = self.net.forward(x)

        if train:
            return all_data
        else:
            return all_data[0][-1]


class DASRHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, perceptual=None,
                 encoder_pretrain_epochs=100, scheduler=None, scheduler_params=None, mean_sub_style=False,
                 ignore_encoder=False, **kwargs):
        super(DASRHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                          **kwargs)
        self.net = DASRPipeline(scale=scale, mean_sub_style=mean_sub_style, **kwargs)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)
        self.contrast_loss = torch.nn.CrossEntropyLoss()
        self.model_name = 'dasr'
        self.encoder_pretrain_epochs = encoder_pretrain_epochs
        self.ignore_encoder = ignore_encoder

    def run_train(self, x, y, tag=None, mask=None, *args, **kwargs):

        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)
        x, y = x.to(device=self.device), y.to(device=self.device)

        if self.ignore_encoder:
            sr, output, _ = self.run_model(x=x[:, 0, ...], x_key=x[:, 1, ...])
            loss_SR = self.criterion(sr, y[:, 0, ...])
            loss_contrast = torch.tensor(0)
            loss = loss_SR
        else:
            if self.curr_epoch < self.encoder_pretrain_epochs:
                _, output, target = self.net.E(im_q=x[:, 0, ...], im_k=x[:, 1, ...])
                target = target.to(device=self.device)
                loss_contrast = self.contrast_loss(output, target)
                loss = loss_contrast
                loss_SR = torch.tensor(0)
            else:
                sr, output, target = self.run_model(x=x[:, 0, ...], x_key=x[:, 1, ...])
                target = target.to(device=self.device)
                loss_SR = self.criterion(sr, y[:, 0, ...])
                loss_contrast = self.contrast_loss(output, target)
                loss = loss_contrast + loss_SR

        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0  TODO: remove duplication?
        loss.backward()  # backpropagate to compute gradients for current iter loss
        if self.grad_clip is not None:  # gradient clipping
            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()  # update network parameters

        loss_package = {}
        for _loss, name in zip((loss, loss_SR, loss_contrast),
                               ('train-loss', 'l1-loss', 'contrast-loss')):
            loss_package[name] = _loss.cpu().data.numpy()

        return loss_package, output.detach().cpu()

    def run_model(self, x, x_key=None, *args, **kwargs):
        if x_key is not None:
            return self.net.forward(x, x_key=x_key)
        else:
            kwargs['model_save_dir'] = self.model_save_dir
            return self.net.forward(x, **kwargs)

    def epoch_end_calls(self):  # only call learning rate scheduler once per epoch
        if self.learning_rate_scheduler is not None:
            self.learning_rate_scheduler.step()


class ContrastiveBlindMetaBedHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, perceptual=None,
                 scheduler=None, scheduler_params=None, selective_meta_blocks='front_only', meta_block='q-layer',
                 encoder_type='default', encoder_output_size=256, pre_trained_encoder_weights=None,
                 embedding_type='pre-q', encoder_freeze_mode='all', **kwargs):
        super(ContrastiveBlindMetaBedHandler, self).__init__(device=device, model_save_dir=model_save_dir,
                                                             eval_mode=eval_mode,
                                                             **kwargs)

        if selective_meta_blocks == 'front_only':
            selective_meta_blocks = [True, False, False, False, False, False, False, False]
        elif selective_meta_blocks == 'none':
            selective_meta_blocks = None

        sr_net = Metabed(scale=scale,
                         num_blocks=8,
                         meta_block=meta_block,
                         selective_meta_blocks=selective_meta_blocks,
                         input_para=encoder_output_size,
                         **kwargs)  # most params are hard-coded for now

        kwargs['model_save_dir'] = model_save_dir

        self.net = ContrastiveBlindSRPipeline(device=device,
                                              generator=sr_net,
                                              encoder=encoder_type,
                                              pre_trained_encoder_weights=pre_trained_encoder_weights,
                                              embedding_type=embedding_type,
                                              encoder_freeze_mode=encoder_freeze_mode)

        self.model_name = 'blind_metabed'
        self.encoder_type = encoder_type

        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)

    def run_model(self, x, *args, **kwargs):
        return self.net.forward(x)

    def extra_diagnostics(self):
        if not self.eval_mode:
            models = [self.net.G, self.net.E]
            model_names = ['Generator-Metabed', 'Encoder-' + self.encoder_type]
            self.print_parameters_model_list(models, model_names)


class ContrastiveBlindQRCANHandler(BaseContrastive):
    def __init__(self, device, model_save_dir,
                 eval_mode=False, lr=1e-4, scale=4, in_features=3,
                 include_sft_layer=False, srmd_mode=False,
                 scheduler=None, scheduler_params=None, style='modulate', perceptual=None, n_feats=64,
                 encoder_type='default', encoder_output_size=256, pre_trained_encoder_weights=None,
                 auxiliary_encoder_weights=None, staggered_encoding=False,
                 embedding_type='pre-q', encoder_freeze_mode='all', encoder_train_eval='eval',
                 combined_loss_mode=None,
                 crop_count=None, data_type='noise',
                 reducer_layer_sizes=None,  # [256, 128, 64, 32, 1]
                 labelling_strategy='triple_precision',
                 **kwargs):
        super(ContrastiveBlindQRCANHandler, self).__init__(device=device,
                                                           model_save_dir=model_save_dir,
                                                           eval_mode=eval_mode,
                                                           labelling_strategy=labelling_strategy,
                                                           **kwargs)

        self.data_type = data_type
        self.crop_count = crop_count

        self.encoder_train_eval = encoder_train_eval

        if reducer_layer_sizes is not None:
            encoder_output_size = reducer_layer_sizes[-1]

        if srmd_mode:
            in_features = in_features + encoder_output_size

        sr_net = QRCAN(scale=scale,
                       in_feats=in_features,
                       num_metadata=encoder_output_size,
                       n_feats=n_feats,
                       style=style,
                       include_sft_layer=include_sft_layer,
                       staggered_encoding=staggered_encoding,
                       **kwargs)

        # combined_loss_mode
        # For now, choose between, None, 'moco', 'supmoco', 'nonblind'

        kwargs['model_save_dir'] = model_save_dir

        self.net = ContrastiveBlindSRPipeline(device=device,
                                              eval_mode=eval_mode,
                                              generator=sr_net,
                                              encoder=encoder_type,
                                              pre_trained_encoder_weights=pre_trained_encoder_weights,
                                              auxiliary_encoder_weights=auxiliary_encoder_weights,
                                              embedding_type=embedding_type,
                                              encoder_freeze_mode=encoder_freeze_mode,
                                              combined_loss_mode=combined_loss_mode,
                                              staggered_encoding=staggered_encoding,
                                              sft_mode=include_sft_layer,
                                              srmd_mode=srmd_mode,
                                              crop_count=crop_count,
                                              reducer_layer_sizes=reducer_layer_sizes,
                                              **kwargs)

        self.model_name = 'blind_qrcan'
        self.encoder_type = encoder_type

        self.combined_loss_mode = combined_loss_mode
        self.contrast_loss = torch.nn.CrossEntropyLoss()

        self.colorspace = 'augmented_rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)

    def run_train(self, x, y, tag=None, mask=None, keep_on_device=False, *args, **kwargs):
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()
        if self.encoder_train_eval == 'eval':
            self.net.E.eval()
            if self.net.aux_E:
                self.net.aux_E.eval()

        if self.crop_count is not None:
            x = torch.cat([x[:, i, :, :, :].squeeze(1) for i in range(x.shape[1])], dim=1)
            y = torch.cat([y[:, i, :, :, :].squeeze(1) for i in range(y.shape[1])], dim=1)
            x, y = x.to(device=self.device), y.to(device=self.device)

        if self.combined_loss_mode is None:
            return super().run_train(x, y, tag, mask, keep_on_device, *args, **kwargs)
        elif self.combined_loss_mode == 'moco':
            sr, output, target = self.net.forward(x[:, 0:3, ...], x[:, 3:, ...])
            target = target.to(device=self.device)

            loss_contrast = self.contrast_loss(output, target)
            loss_SR = self.criterion(sr, y[:, 0:3, ...])

            loss = loss_contrast + loss_SR

            self.standard_update(loss)

            loss_package = {}
            for _loss, name in zip((loss, loss_SR, loss_contrast), ('train-loss', 'l1-loss', 'contrast-loss')):
                loss_package[name] = _loss.cpu().data.numpy()

            return loss_package, output.detach().cpu()
        elif self.combined_loss_mode == 'supmoco':
            labels = self.class_logic(kwargs['metadata'], kwargs['metadata_keys'])
            self.net.E.set_class_count(self.total_classes)
            self.num_classes = self.net.E.num_classes

            x = x.view(-1, 3, x.shape[2], x.shape[3]).to(device=self.device)
            y = y.view(-1, 3, y.shape[2], y.shape[3]).to(device=self.device)

            batch_count = int(x.shape[0] / self.crop_count)
            indices = [i * self.crop_count for i in range(batch_count)]
            non_indices = [i for i in range(x.shape[0]) if i not in indices]

            sr, output, full_labels = self.net.forward(x[indices], x[non_indices], labels.squeeze())
            full_labels = full_labels.to(self.device)

            # apply cross-entropy loss (final part of supmoco loss)
            loss_contrast = self.contrast_loss(output, full_labels)
            loss_SR = self.criterion(sr, y[indices])

            loss = loss_contrast + loss_SR

            self.standard_update(loss)

            loss_package = {}
            for _loss, name in zip((loss, loss_SR, loss_contrast), ('train-loss', 'l1-loss', 'contrast-loss')):
                loss_package[name] = _loss.cpu().data.numpy()

            return loss_package, output.detach().cpu()
        elif self.combined_loss_mode == 'nonblind':
            x, y = x.to(device=self.device), y.to(device=self.device)
            sr, reduced_embedding = self.run_model(x, *args, **kwargs)

            reduced_embedding = reduced_embedding.squeeze(3).squeeze(2)

            # TODO: find better way to do this
            key_index = [keys[0] for keys in kwargs['metadata_keys']].index('realesrganblur-sigma_x')
            selected_metadata = kwargs['metadata'][:, key_index].unsqueeze(1).to(device=self.device)

            loss_nonblind = self.criterion(reduced_embedding, selected_metadata)
            loss_SR = self.criterion(sr, y)

            loss = loss_nonblind + loss_SR

            self.standard_update(loss)

            loss_package = {}
            for _loss, name in zip((loss, loss_SR, loss_nonblind), ('train-loss', 'l1-loss', 'nonblind-loss')):
                loss_package[name] = _loss.cpu().data.numpy()

            return loss_package, sr.detach().cpu()

    def run_model(self, x, *args, **kwargs):
        return self.net.forward(x, **kwargs)

    def extra_diagnostics(self):
        if not self.eval_mode:
            models = [self.net.G, self.net.E]
            model_names = ['Generator-QRCAN', 'Encoder-' + self.encoder_type]

            if self.net.reducer is not None:
                models = models + [self.net.reducer]
                model_names = model_names + ['Reducer']

            self.print_parameters_model_list(models, model_names)


class ContrastiveBlindQHANHandler(BaseContrastive):
    def __init__(self, device, model_save_dir,
                 eval_mode=False, lr=1e-4, scale=4, in_features=3,
                 include_sft_layer=False, srmd_mode=False,
                 scheduler=None, scheduler_params=None, style='modulate', perceptual=None, n_feats=64,
                 encoder_type='default', encoder_output_size=256, pre_trained_encoder_weights=None,
                 auxiliary_encoder_weights=None, staggered_encoding=False,
                 embedding_type='pre-q', encoder_freeze_mode='all', encoder_train_eval='eval',
                 combined_loss_mode=None,
                 crop_count=None, data_type='noise',
                 reducer_layer_sizes=None,  # [256, 128, 64, 32, 1]
                 **kwargs):
        super(ContrastiveBlindQHANHandler, self).__init__(device=device,
                                                          model_save_dir=model_save_dir,
                                                          eval_mode=eval_mode,
                                                          **kwargs)

        self.data_type = data_type
        self.crop_count = crop_count

        self.encoder_train_eval = encoder_train_eval

        if reducer_layer_sizes is not None:
            encoder_output_size = reducer_layer_sizes[-1]

        if srmd_mode:
            in_features = in_features + encoder_output_size

        sr_net = QHAN(scale=scale,
                      in_feats=in_features,
                      num_metadata=encoder_output_size,
                      n_feats=n_feats,
                      style=style,
                      include_sft_layer=include_sft_layer,
                      staggered_encoding=staggered_encoding,
                      **kwargs)

        # combined_loss_mode
        # For now, choose between, None, 'moco', 'supmoco', 'nonblind'

        kwargs['model_save_dir'] = model_save_dir

        self.net = ContrastiveBlindSRPipeline(device=device,
                                              eval_mode=eval_mode,
                                              generator=sr_net,
                                              encoder=encoder_type,
                                              pre_trained_encoder_weights=pre_trained_encoder_weights,
                                              auxiliary_encoder_weights=auxiliary_encoder_weights,
                                              embedding_type=embedding_type,
                                              encoder_freeze_mode=encoder_freeze_mode,
                                              combined_loss_mode=combined_loss_mode,
                                              staggered_encoding=staggered_encoding,
                                              sft_mode=include_sft_layer,
                                              srmd_mode=srmd_mode,
                                              crop_count=crop_count,
                                              reducer_layer_sizes=reducer_layer_sizes,
                                              **kwargs)

        self.model_name = 'blind_qhan'
        self.encoder_type = encoder_type

        self.combined_loss_mode = combined_loss_mode
        self.contrast_loss = torch.nn.CrossEntropyLoss()

        self.colorspace = 'augmented_rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)

    def run_train(self, x, y, tag=None, mask=None, keep_on_device=False, *args, **kwargs):
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()
        if self.encoder_train_eval == 'eval':
            self.net.E.eval()
            if self.net.aux_E:
                self.net.aux_E.eval()

        if self.crop_count is not None:
            x = torch.cat([x[:, i, :, :, :].squeeze(1) for i in range(x.shape[1])], dim=1)
            y = torch.cat([y[:, i, :, :, :].squeeze(1) for i in range(y.shape[1])], dim=1)
            x, y = x.to(device=self.device), y.to(device=self.device)

        if self.combined_loss_mode is None:
            return super().run_train(x, y, tag, mask, keep_on_device, *args, **kwargs)
        elif self.combined_loss_mode == 'moco':
            sr, output, target = self.net.forward(x[:, 0:3, ...], x[:, 3:, ...])
            target = target.to(device=self.device)

            loss_contrast = self.contrast_loss(output, target)
            loss_SR = self.criterion(sr, y[:, 0:3, ...])

            loss = loss_contrast + loss_SR

            self.standard_update(loss)

            loss_package = {}
            for _loss, name in zip((loss, loss_SR, loss_contrast), ('train-loss', 'l1-loss', 'contrast-loss')):
                loss_package[name] = _loss.cpu().data.numpy()

            return loss_package, output.detach().cpu()
        elif self.combined_loss_mode == 'supmoco':
            labels = self.class_logic(kwargs['metadata'], kwargs['metadata_keys'])
            self.net.E.set_class_count(self.total_classes)
            self.num_classes = self.net.E.num_classes

            x = x.view(-1, 3, x.shape[2], x.shape[3]).to(device=self.device)
            y = y.view(-1, 3, y.shape[2], y.shape[3]).to(device=self.device)

            batch_count = int(x.shape[0] / self.crop_count)
            indices = [i * self.crop_count for i in range(batch_count)]
            non_indices = [i for i in range(x.shape[0]) if i not in indices]

            sr, output, full_labels = self.net.forward(x[indices], x[non_indices], labels.squeeze())
            full_labels = full_labels.to(self.device)

            # apply cross-entropy loss (final part of supmoco loss)
            loss_contrast = self.contrast_loss(output, full_labels)
            loss_SR = self.criterion(sr, y[indices])

            loss = loss_contrast + loss_SR

            self.standard_update(loss)

            loss_package = {}
            for _loss, name in zip((loss, loss_SR, loss_contrast), ('train-loss', 'l1-loss', 'contrast-loss')):
                loss_package[name] = _loss.cpu().data.numpy()

            return loss_package, output.detach().cpu()
        elif self.combined_loss_mode == 'nonblind':
            x, y = x.to(device=self.device), y.to(device=self.device)
            sr, reduced_embedding = self.run_model(x, *args, **kwargs)

            reduced_embedding = reduced_embedding.squeeze(3).squeeze(2)

            # TODO: find better way to do this
            key_index = [keys[0] for keys in kwargs['metadata_keys']].index('realesrganblur-sigma_x')
            selected_metadata = kwargs['metadata'][:, key_index].unsqueeze(1).to(device=self.device)

            loss_nonblind = self.criterion(reduced_embedding, selected_metadata)
            loss_SR = self.criterion(sr, y)

            loss = loss_nonblind + loss_SR

            self.standard_update(loss)

            loss_package = {}
            for _loss, name in zip((loss, loss_SR, loss_nonblind), ('train-loss', 'l1-loss', 'nonblind-loss')):
                loss_package[name] = _loss.cpu().data.numpy()

            return loss_package, sr.detach().cpu()

    def run_model(self, x, *args, **kwargs):
        return self.net.forward(x, **kwargs)

    def extra_diagnostics(self):
        if not self.eval_mode:
            models = [self.net.G, self.net.E]
            model_names = ['Generator-QHAN', 'Encoder-' + self.encoder_type]

            if self.net.reducer is not None:
                models = models + [self.net.reducer]
                model_names = model_names + ['Reducer']

            self.print_parameters_model_list(models, model_names)


class ContrastiveBlindQEDSRHandler(BaseContrastive):
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, in_features=3, num_blocks=16,
                 num_features=64, res_scale=0.1, scheduler=None, scheduler_params=None,
                 perceptual=None, encoder_type='default', encoder_output_size=256, pre_trained_encoder_weights=None,
                 embedding_type='pre-q', encoder_freeze_mode='all', encoder_train_eval='eval', combined_loss_mode=None,
                 crop_count=None, data_type='noise',
                 max_combined_im_size=160000,
                 **kwargs):
        super(ContrastiveBlindQEDSRHandler, self).__init__(device=device,
                                                           model_save_dir=model_save_dir,
                                                           eval_mode=eval_mode,
                                                           **kwargs)

        self.data_type = data_type
        self.crop_count = crop_count

        self.encoder_train_eval = encoder_train_eval

        self.max_combined_im_size = max_combined_im_size
        self.scale = scale

        sr_net = QEDSR(scale=scale,
                       in_features=in_features,
                       num_features=num_features,
                       num_blocks=num_blocks,
                       res_scale=res_scale,
                       input_para=encoder_output_size,
                       **kwargs)

        # combined_loss_mode
        # For now, choose between, None, 'moco' or 'supmoco'

        kwargs['model_save_dir'] = model_save_dir

        self.net = ContrastiveBlindSRPipeline(device=device,
                                              eval_mode=eval_mode,
                                              generator=sr_net,
                                              encoder=encoder_type,
                                              pre_trained_encoder_weights=pre_trained_encoder_weights,
                                              embedding_type=embedding_type,
                                              encoder_freeze_mode=encoder_freeze_mode,
                                              combined_loss_mode=combined_loss_mode,
                                              crop_count=crop_count)

        self.model_name = 'blind_qedsr'
        self.encoder_type = encoder_type

        self.combined_loss_mode = combined_loss_mode
        self.contrast_loss = torch.nn.CrossEntropyLoss()

        self.colorspace = 'augmented_rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)

    def run_train(self, x, y, tag=None, mask=None, keep_on_device=False, *args, **kwargs):
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()
        if self.encoder_train_eval == 'eval':
            self.net.E.eval()

        if self.crop_count is not None:
            x = torch.cat([x[:, i, :, :, :].squeeze(1) for i in range(x.shape[1])], dim=1)
            y = torch.cat([y[:, i, :, :, :].squeeze(1) for i in range(y.shape[1])], dim=1)
            x, y = x.to(device=self.device), y.to(device=self.device)

        if self.combined_loss_mode is None:
            return super().run_train(x, y, tag, mask, keep_on_device, *args, **kwargs)
        elif self.combined_loss_mode == 'moco':
            sr, output, target = self.net.forward(x[:, 0:3, ...], x[:, 3:, ...])
            target = target.to(device=self.device)

            loss_contrast = self.contrast_loss(output, target)
            loss_SR = self.criterion(sr, y[:, 0:3, ...])

            loss = loss_contrast + loss_SR

            self.standard_update(loss)

            loss_package = {}
            for _loss, name in zip((loss, loss_SR, loss_contrast), ('train-loss', 'l1-loss', 'contrast-loss')):
                loss_package[name] = _loss.cpu().data.numpy()

            return loss_package, output.detach().cpu()
        elif self.combined_loss_mode == 'supmoco':
            labels = self.class_logic(kwargs['metadata'], kwargs['metadata_keys'])
            self.net.E.set_class_count(self.total_classes)
            self.num_classes = self.net.E.num_classes

            x = x.view(-1, 3, x.shape[2], x.shape[3]).to(device=self.device)
            y = y.view(-1, 3, y.shape[2], y.shape[3]).to(device=self.device)

            batch_count = int(x.shape[0] / self.crop_count)
            indices = [i * self.crop_count for i in range(batch_count)]
            non_indices = [i for i in range(x.shape[0]) if i not in indices]

            sr, output, full_labels = self.net.forward(x[indices], x[non_indices], labels.squeeze())
            full_labels = full_labels.to(self.device)

            # apply cross-entropy loss (final part of supmoco loss)
            loss_contrast = self.contrast_loss(output, full_labels)
            loss_SR = self.criterion(sr, y[indices])

            loss = loss_contrast + loss_SR

            self.standard_update(loss)

            loss_package = {}
            for _loss, name in zip((loss, loss_SR, loss_contrast), ('train-loss', 'l1-loss', 'contrast-loss')):
                loss_package[name] = _loss.cpu().data.numpy()

            return loss_package, output.detach().cpu()

    def run_model(self, x, *args, **kwargs):
        return self.net.forward(x)

    def forward_chop(self, x, shave=10):
        # modified from https://github.com/daitao/SAN/blob/master/TestCode/code/model/__init__.py

        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave

        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < self.max_combined_im_size:
            sr_list = []
            for chunk in lr_list:
                sr_list.append(self.run_chopped_eval(chunk))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave)
                for patch in lr_list]

        h, w = self.scale * h, self.scale * w
        h_half, w_half = self.scale * h_half, self.scale * w_half
        h_size, w_size = self.scale * h_size, self.scale * w_size
        shave *= self.scale

        output = x.new(b, c, h, w)

        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def run_eval(self, x, y=None, request_loss=False, tag=None, timing=False, keep_on_device=False, *args, **kwargs):
        if timing:
            tic = time.perf_counter()
        sr_image = self.forward_chop(x)
        if timing:
            toc = time.perf_counter()

        if request_loss:
            return sr_image, self.criterion(sr_image, y), toc - tic if timing else None
        else:
            return sr_image, None, toc - tic if timing else None

    def run_chopped_eval(self, x):
        return super().run_eval(x)[0]

    def extra_diagnostics(self):
        if not self.eval_mode:
            models = [self.net.G, self.net.E]
            model_names = ['Generator-QEDSR', 'Encoder-' + self.encoder_type]
            self.print_parameters_model_list(models, model_names)


class ContrastiveBlindQRealESRGANHandler(QRealESRGANHandler, BaseContrastive):
    def __init__(self, device, model_save_dir, eval_mode=False,
                 encoder_type='default', encoder_output_size=256, pre_trained_encoder_weights=None,
                 embedding_type='pre-q', encoder_freeze_mode='all', encoder_train_eval='eval',
                 **kwargs):
        super(ContrastiveBlindQRealESRGANHandler, self).__init__(device=device,
                                                                 model_save_dir=model_save_dir,
                                                                 eval_mode=eval_mode,
                                                                 metadata_bypass_len=encoder_output_size,
                                                                 **kwargs)

        kwargs['model_save_dir'] = model_save_dir

        self.net = ContrastiveBlindSRPipeline(device=device,
                                              eval_mode=eval_mode,
                                              generator=self.net,
                                              encoder=encoder_type,
                                              pre_trained_encoder_weights=pre_trained_encoder_weights,
                                              embedding_type=embedding_type,
                                              encoder_freeze_mode=encoder_freeze_mode,
                                              **kwargs)
        self.encoder_train_eval = encoder_train_eval
        self.model_name = 'blind_qrealesrgan'
        self.activate_device()

    def run_train(self, x, y, metadata, extra_channels=None, metadata_keys=None, *args, **kwargs):
        """
        Runs one training iteration (pre-train or GAN) through a data batch
        :param x: input
        :param y: target
        :return: calculated loss pre-backprop, output image
        """
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)

        if self.encoder_train_eval == 'eval':
            self.net.E.eval()

        self.discriminator.train()

        x, y = x.to(device=self.device), y.to(device=self.device)

        if self.curr_epoch < self.pretrain_epochs:  # L1 pre-training
            out = self.run_model(x)  # run data through model
            loss = self.criterion(out, y)  # compute L1 loss
            self.pre_train_update(loss)
            loss_G = loss
            loss_content = torch.tensor(0.0, device=self.device)
            loss_GAN = torch.tensor(0.0, device=self.device)
            loss_real = torch.tensor(0.0, device=self.device)
            loss_fake = torch.tensor(0.0, device=self.device)
        else:
            out = self.run_model(x)  # run data through model
            loss_G, loss_content, loss_GAN, loss = self.generator_update(out, y)
            loss_real, loss_fake = self.discriminator_update(out, y)

        loss_package = {}

        for _loss, name in zip((loss_G, loss, loss_GAN, loss_content, loss_real, loss_fake),
                               ('train-loss', 'l1-loss', 'gan-loss', 'vgg-loss', 'd-loss-real', 'd-loss-fake')):
            loss_package[name] = _loss.cpu().data.numpy()

        return loss_package, out.detach().cpu()

    def run_model(self, x, *args, **kwargs):
        return self.net.forward(x)

    def run_eval(self, **kwargs):
        return super(BaseContrastive, self).run_eval(**kwargs)


class IKCPredictorHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, perceptual=None,
                 optimizer_params=None, scheduler=None, scheduler_params=None, **kwargs):
        super(IKCPredictorHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                                  **kwargs)
        self.net = Predictor(**kwargs)
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device, optimizer_params=optimizer_params)
        self.criterion = nn.MSELoss()
        self.model_name = 'ikcpredictor'


class IKCCorrectorHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, perceptual=None,
                 optimizer_params=None, scheduler=None, scheduler_params=None, **kwargs):
        super(IKCCorrectorHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                                  **kwargs)
        self.net = Corrector(**kwargs)
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device, optimizer_params=optimizer_params)
        self.criterion = nn.MSELoss()
        self.model_name = 'ikccorrector'

    def run_model(self, x, extra_channels=None, **kwargs):
        return self.net.forward(x, extra_channels)


class IKC(MultiModel):
    """
    IKC model based on code from https://github.com/yuanjunchai/IKC

    Note that final iteration from IKC loop is not necessarily the best image.
    Any of the images from the corrector iterations (typically 7) could have the best PSNR.
    This handler makes sure to select the best image from the entire loop.  This is not necessarily best practise, however,
    as in a real scenario, you won't have a ground truth reference point.
    Nevertheless, this is the method pitched in the original paper/code.
    """

    def __init__(self, sftmd_pretrain_epochs=None, correction_steps=None, pre_trained_model_details=None,
                 force_final_eval_iter=True, **kwargs):
        super(IKC, self).__init__(**kwargs)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.sftmd_pretrain_epochs = sftmd_pretrain_epochs
        self.model_name = 'ikc'
        self.force_final_eval_iter = force_final_eval_iter

        self.correction_steps = correction_steps
        sftmd_loss_package = {'sftmd_loss_%d' % it: np.array(0) for it in range(correction_steps)}
        corrector_loss_package = {'corrector_loss_%d' % it: np.array(0) for it in range(correction_steps)}
        self.empty_loss_package = {**sftmd_loss_package, **corrector_loss_package, 'predictor-loss': np.array(0)}
        self.pre_trained_model_details = pre_trained_model_details

    def run_train(self, x, y, tag=None, mask=None, extra_channels=None, metadata=None,
                  metadata_keys=None, *args, **kwargs):
        if self.child_models['sr_model'].curr_epoch < self.sftmd_pretrain_epochs:
            loss_sftmd, output = self.child_models['sr_model'].run_train(x=x, y=y, metadata=metadata,
                                                                         metadata_keys=metadata_keys, **kwargs)
            loss_package = {'train-loss': loss_sftmd, **self.empty_loss_package}
            return loss_package, output
        else:
            loss_package = {}
            metadata_code = \
                self.child_models['sr_model'].generate_channels(x, vector_override=True,
                                                                metadata=metadata,
                                                                metadata_keys=metadata_keys)
            metadata_code = metadata_code.squeeze().to(self.device)

            p_loss, predicted_kernel = self.child_models['predictor'].run_train(x=x, y=metadata_code,
                                                                                keep_on_device=True, **kwargs)

            loss_package['predictor-loss'] = p_loss

            # TODO: metadata_key system is very hacky, really needs to change!
            ikc_package, final_image, best_loss = self.ikc_loop(x, y, kernel=predicted_kernel, request_loss=True,
                                                                real_kernel=metadata_code,
                                                                metadata_keys=[
                                                                    ['blur_kernel'] * predicted_kernel.shape[1]],
                                                                **kwargs)

            loss_package = {**loss_package, **ikc_package, 'train-loss': best_loss}

            return loss_package, final_image

    def run_eval(self, x, y=None, request_loss=False, tag=None, metadata=None, metadata_keys=None, timing=False,
                 *args, **kwargs):

        if self.child_models['sr_model'].curr_epoch < self.sftmd_pretrain_epochs:
            output, loss, runtime = self.child_models['sr_model'].run_eval(x=x, y=y, timing=timing,
                                                                           request_loss=request_loss,
                                                                           metadata=metadata,
                                                                           metadata_keys=metadata_keys, **kwargs)
            return output, loss, runtime
        else:
            if timing:
                tic = time.perf_counter()
            predicted_kernel, _, _ = self.child_models['predictor'].run_eval(x=x, request_loss=False, timing=False,
                                                                             keep_on_device=True,
                                                                             **kwargs)

            kwargs['metadata_keys_non_blur'] = metadata_keys
            kwargs['metadata_non_blur'] = metadata

            ikc_package, final_image, best_loss = self.ikc_loop(x, y, kernel=predicted_kernel,
                                                                metadata_keys=[
                                                                    ['blur_kernel'] * predicted_kernel.shape[1]],
                                                                training=False, request_loss=request_loss, **kwargs)
            if timing:
                toc = time.perf_counter()
            return final_image, best_loss, toc - tic if timing else None

    def ikc_loop(self, x, y, kernel, real_kernel=None, training=True, request_loss=False, **kwargs):

        predicted_kernel = kernel
        loss_package = {}
        sftmd_losses = []
        all_images = []

        for step in range(self.correction_steps):

            # run SFTMD in eval mode for corresponding SR image
            out_rgb, loss_sftmd, _ = self.child_models['sr_model'].run_eval(x=x, y=y, metadata=predicted_kernel,
                                                                            request_loss=True, keep_on_device=True,
                                                                            timing=False, **kwargs)
            if training:
                c_loss, predicted_kernel = self.child_models['corrector'].run_train(x=out_rgb,
                                                                                    extra_channels=predicted_kernel,
                                                                                    request_loss=request_loss,
                                                                                    keep_on_device=True,
                                                                                    y=real_kernel, **kwargs)
            else:
                predicted_kernel, c_loss, _ = self.child_models['corrector'].run_eval(x=out_rgb, request_loss=False,
                                                                                      extra_channels=predicted_kernel,
                                                                                      keep_on_device=True,
                                                                                      y=real_kernel, timing=False,
                                                                                      **kwargs)
            loss_package['sftmd_loss_%d' % step] = loss_sftmd
            loss_package['corrector_loss_%d' % step] = c_loss
            sftmd_losses.append(loss_sftmd)
            all_images.append(out_rgb.cpu())

        norm_path = os.path.normpath(self.model_save_dir)
        split_path = norm_path.split(os.sep)

        if loss_sftmd is not None and not self.force_final_eval_iter:
            # best image (dependant on loss - which requires a corresponding GT image to be available) is extracted
            best_loss = min(sftmd_losses)
            best_image = all_images[sftmd_losses.index(best_loss)]
        else:  # if no GT image provided, then must default to last image iteration as best result.
            best_loss = sftmd_losses[-1]
            best_image = all_images[-1]

        return loss_package, best_image, best_loss

    def set_epoch(self, epoch):
        self.curr_epoch = epoch
        for key, model in self.child_models.items():
            if key == 'sr_model' and model.curr_epoch >= self.sftmd_pretrain_epochs:
                continue  # SFTMD will no longer be updated after the setpoint
            else:
                model.set_epoch(epoch)

    def pre_training_model_load(self):
        if self.pre_trained_model_details is not None:
            self.child_models[self.pre_trained_model_details['name']].load_model('train_model',
                                                                                 self.pre_trained_model_details[
                                                                                     'epoch'], True,
                                                                                 load_override=os.path.join(
                                                                                     self.pre_trained_model_details[
                                                                                         'loc'], 'saved_models'))

    @staticmethod
    def best_model_selection_criteria(log_dir=None, log_file='summary.csv', model_metadata=None,
                                      stats=None, stats_dir=None, base_metric='val-PSNR'):
        """
        IKC has a pre-training phase.  This function ignores this phase when
        finding the best epoch based on the selected metric.
        """

        if stats_dir and stats is None:
            stats = load_statistics(log_dir, log_file, config='pd')  # loads model training stats if not provided

        if not model_metadata:
            model_metadata = toml.load(os.path.join(os.path.dirname(log_dir), 'config.toml'))['model']

        cutoff_epoch = model_metadata['sftmd_pretrain_epochs']

        load_epoch = standard_metric_epoch_selection(base_metric, stats[cutoff_epoch:])
        return load_epoch


class ContrastiveBlindQELANHandler(BaseContrastive):
    def __init__(self, device, model_save_dir,
                 eval_mode=False, lr=1e-4, scale=4, in_features=3,
                 include_sft_layer=False, srmd_mode=False,
                 scheduler=None, scheduler_params=None, perceptual=None,
                 encoder_type='default', encoder_output_size=256, pre_trained_encoder_weights=None,
                 auxiliary_encoder_weights=None, staggered_encoding=False,
                 embedding_type='pre-q', encoder_freeze_mode='all', encoder_train_eval='eval',
                 combined_loss_mode=None,
                 crop_count=None, data_type='noise',
                 **kwargs):
        super(ContrastiveBlindQELANHandler, self).__init__(device=device,
                                                           model_save_dir=model_save_dir,
                                                           eval_mode=eval_mode,
                                                           **kwargs)

        self.data_type = data_type
        self.crop_count = crop_count

        self.encoder_train_eval = encoder_train_eval

        if srmd_mode:
            in_features = in_features + encoder_output_size

        sr_net = QELAN(scale=scale, colors=in_features, num_metadata=encoder_output_size, **kwargs)

        # combined_loss_mode
        # For now, choose between, None, 'moco' or 'supmoco'

        self.net = ContrastiveBlindSRPipeline(device=device,
                                              eval_mode=eval_mode,
                                              generator=sr_net,
                                              encoder=encoder_type,
                                              pre_trained_encoder_weights=pre_trained_encoder_weights,
                                              auxiliary_encoder_weights=auxiliary_encoder_weights,
                                              embedding_type=embedding_type,
                                              encoder_freeze_mode=encoder_freeze_mode,
                                              combined_loss_mode=combined_loss_mode,
                                              staggered_encoding=staggered_encoding,
                                              sft_mode=include_sft_layer,
                                              srmd_mode=srmd_mode,
                                              crop_count=crop_count, **kwargs)

        self.model_name = 'blind_qelan'
        self.encoder_type = encoder_type

        self.combined_loss_mode = combined_loss_mode
        self.contrast_loss = torch.nn.CrossEntropyLoss()

        self.colorspace = 'augmented_rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)
        if scheduler == 'multi_step_lr':
            self.end_epoch_scheduler = True
        else:
            self.end_epoch_scheduler = False

    def run_train(self, x, y, tag=None, mask=None, keep_on_device=False, *args, **kwargs):
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()
        if self.encoder_train_eval == 'eval':
            self.net.E.eval()
            if self.net.aux_E:
                self.net.aux_E.eval()

        if self.crop_count is not None:
            x = torch.cat([x[:, i, :, :, :].squeeze(1) for i in range(x.shape[1])], dim=1)
            y = torch.cat([y[:, i, :, :, :].squeeze(1) for i in range(y.shape[1])], dim=1)
            x, y = x.to(device=self.device), y.to(device=self.device)

        if self.combined_loss_mode is None:
            return super().run_train(x, y, tag, mask, keep_on_device, scheduler_skip=self.end_epoch_scheduler, *args, **kwargs)
        elif self.combined_loss_mode == 'moco':
            sr, output, target = self.net.forward(x[:, 0:3, ...], x[:, 3:, ...])
            target = target.to(device=self.device)

            loss_contrast = self.contrast_loss(output, target)
            loss_SR = self.criterion(sr, y[:, 0:3, ...])

            loss = loss_contrast + loss_SR

            self.standard_update(loss, scheduler_skip=self.end_epoch_scheduler)

            loss_package = {}
            for _loss, name in zip((loss, loss_SR, loss_contrast), ('train-loss', 'l1-loss', 'contrast-loss')):
                loss_package[name] = _loss.cpu().data.numpy()

            return loss_package, output.detach().cpu()
        elif self.combined_loss_mode == 'supmoco':
            labels = self.class_logic(kwargs['metadata'], kwargs['metadata_keys'])
            self.net.E.register_classes(self.total_classes)
            self.num_classes = self.net.E.num_classes

            x = x.view(-1, 3, x.shape[2], x.shape[3]).to(device=self.device)
            y = y.view(-1, 3, y.shape[2], y.shape[3]).to(device=self.device)

            batch_count = int(x.shape[0] / self.crop_count)
            indices = [i * self.crop_count for i in range(batch_count)]
            non_indices = [i for i in range(x.shape[0]) if i not in indices]

            sr, output, full_labels = self.net.forward(x[indices], x[non_indices], labels.squeeze())
            full_labels = full_labels.to(self.device)

            # apply cross-entropy loss (final part of supmoco loss)
            loss_contrast = self.contrast_loss(output, full_labels)
            loss_SR = self.criterion(sr, y[indices])

            loss = loss_contrast + loss_SR

            self.standard_update(loss, scheduler_skip=self.end_epoch_scheduler)

            loss_package = {}
            for _loss, name in zip((loss, loss_SR, loss_contrast), ('train-loss', 'l1-loss', 'contrast-loss')):
                loss_package[name] = _loss.cpu().data.numpy()

            return loss_package, output.detach().cpu()

    def run_model(self, x, *args, **kwargs):
        return self.net.forward(x)

    def extra_diagnostics(self):
        if not self.eval_mode:
            models = [self.net.G, self.net.E]
            model_names = ['Generator-QELAN', 'Encoder-' + self.encoder_type]
            self.print_parameters_model_list(models, model_names)

    def epoch_end_calls(self):
        if self.learning_rate_scheduler is not None and isinstance(self.learning_rate_scheduler,
                                                                   torch.optim.lr_scheduler.MultiStepLR):
            self.learning_rate_scheduler.step()


class ContrastiveBlindQSANHandler(BaseContrastive):
    def __init__(self, device, model_save_dir,
                 eval_mode=False, lr=1e-4, scale=4, in_features=3,
                 include_sft_layer=False, srmd_mode=False,
                 scheduler=None, scheduler_params=None, perceptual=None,
                 encoder_type='default', encoder_output_size=256, pre_trained_encoder_weights=None,
                 auxiliary_encoder_weights=None, staggered_encoding=False,
                 embedding_type='pre-q', encoder_freeze_mode='all', encoder_train_eval='eval',
                 combined_loss_mode=None,
                 crop_count=None, data_type='noise',
                 max_combined_im_size=160000,
                 **kwargs):
        super(ContrastiveBlindQSANHandler, self).__init__(device=device,
                                                          model_save_dir=model_save_dir,
                                                          eval_mode=eval_mode,
                                                          **kwargs)

        self.data_type = data_type
        self.crop_count = crop_count
        self.max_combined_im_size = max_combined_im_size
        self.encoder_train_eval = encoder_train_eval
        self.scale = scale

        if srmd_mode:
            in_features = in_features + encoder_output_size

        sr_net = QSAN(scale=scale, n_colors=in_features, input_para=encoder_output_size, **kwargs)

        # combined_loss_mode
        # For now, choose between, None, 'moco' or 'supmoco'

        self.net = ContrastiveBlindSRPipeline(device=device,
                                              eval_mode=eval_mode,
                                              generator=sr_net,
                                              encoder=encoder_type,
                                              pre_trained_encoder_weights=pre_trained_encoder_weights,
                                              auxiliary_encoder_weights=auxiliary_encoder_weights,
                                              embedding_type=embedding_type,
                                              encoder_freeze_mode=encoder_freeze_mode,
                                              combined_loss_mode=combined_loss_mode,
                                              staggered_encoding=staggered_encoding,
                                              sft_mode=include_sft_layer,
                                              srmd_mode=srmd_mode,
                                              crop_count=crop_count, **kwargs)

        self.model_name = 'blind_qsan'
        self.encoder_type = encoder_type

        self.combined_loss_mode = combined_loss_mode
        self.contrast_loss = torch.nn.CrossEntropyLoss()

        self.colorspace = 'augmented_rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)

    def run_train(self, x, y, tag=None, mask=None, keep_on_device=False, *args, **kwargs):
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()
        if self.encoder_train_eval == 'eval':
            self.net.E.eval()
            if self.net.aux_E:
                self.net.aux_E.eval()

        if self.crop_count is not None:
            x = torch.cat([x[:, i, :, :, :].squeeze(1) for i in range(x.shape[1])], dim=1)
            y = torch.cat([y[:, i, :, :, :].squeeze(1) for i in range(y.shape[1])], dim=1)
            x, y = x.to(device=self.device), y.to(device=self.device)

        if self.combined_loss_mode is None:
            return super().run_train(x, y, tag, mask, keep_on_device, *args, **kwargs)
        elif self.combined_loss_mode == 'moco':
            sr, output, target = self.net.forward(x[:, 0:3, ...], x[:, 3:, ...])
            target = target.to(device=self.device)

            loss_contrast = self.contrast_loss(output, target)
            loss_SR = self.criterion(sr, y[:, 0:3, ...])

            loss = loss_contrast + loss_SR

            self.standard_update(loss)

            loss_package = {}
            for _loss, name in zip((loss, loss_SR, loss_contrast), ('train-loss', 'l1-loss', 'contrast-loss')):
                loss_package[name] = _loss.cpu().data.numpy()

            return loss_package, output.detach().cpu()
        elif self.combined_loss_mode == 'supmoco':
            labels = self.class_logic(kwargs['metadata'], kwargs['metadata_keys'])
            self.net.E.register_classes(self.total_classes)
            self.num_classes = self.net.E.num_classes

            x = x.view(-1, 3, x.shape[2], x.shape[3]).to(device=self.device)
            y = y.view(-1, 3, y.shape[2], y.shape[3]).to(device=self.device)

            batch_count = int(x.shape[0] / self.crop_count)
            indices = [i * self.crop_count for i in range(batch_count)]
            non_indices = [i for i in range(x.shape[0]) if i not in indices]

            sr, output, full_labels = self.net.forward(x[indices], x[non_indices], labels.squeeze())
            full_labels = full_labels.to(self.device)

            # apply cross-entropy loss (final part of supmoco loss)
            loss_contrast = self.contrast_loss(output, full_labels)
            loss_SR = self.criterion(sr, y[indices])

            loss = loss_contrast + loss_SR

            self.standard_update(loss)

            loss_package = {}
            for _loss, name in zip((loss, loss_SR, loss_contrast), ('train-loss', 'l1-loss', 'contrast-loss')):
                loss_package[name] = _loss.cpu().data.numpy()

            return loss_package, output.detach().cpu()

    def forward_chop(self, x, shave=10):
        # modified from https://github.com/daitao/SAN/blob/master/TestCode/code/model/__init__.py

        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave

        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < self.max_combined_im_size:
            sr_list = []
            for chunk in lr_list:
                sr_list.append(super().run_eval(chunk, request_loss=False)[0])
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave)
                for patch in lr_list]

        h, w = self.scale * h, self.scale * w
        h_half, w_half = self.scale * h_half, self.scale * w_half
        h_size, w_size = self.scale * h_size, self.scale * w_size
        shave *= self.scale

        output = x.new(b, c, h, w)

        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def run_eval(self, x, y=None, request_loss=False, metadata=None, metadata_keys=None, timing=False, *args, **kwargs):
        if timing:
            tic = time.perf_counter()
        sr_image = self.forward_chop(x)
        if timing:
            toc = time.perf_counter()
        if request_loss:
            return sr_image, self.criterion(sr_image, y), toc - tic if timing else None
        else:
            return sr_image, None, toc - tic if timing else None

    def run_model(self, x, *args, **kwargs):
        return self.net.forward(x)

    def extra_diagnostics(self):
        if not self.eval_mode:
            models = [self.net.G, self.net.E]
            model_names = ['Generator-QSAN', 'Encoder-' + self.encoder_type]
            self.print_parameters_model_list(models, model_names)
