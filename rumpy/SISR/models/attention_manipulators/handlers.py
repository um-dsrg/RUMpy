import time

from rumpy.SISR.models.attention_manipulators import QModel
from rumpy.SISR.models.non_blind_gan_models import BaseBSRGANModel
from rumpy.SISR.models.non_blind_gan_models.handlers import *

from .architectures import *
from .mini_model import Metabed


class QRCANHandler(QModel):
    """
    Meta-modified QRCAN.  Standard setup consists of 10 residual groups, each with 20 residual blocks.

    Meta-attention can be selectively inserted in different layers using the include_q_layer,
    selective_meta_blocks and num_q_layers_inner_residual parameters:
    include_q_layer: Bool; set to True to insert q-layers within network residual blocks.
    selective_meta_blocks: List of Bools; must be the same length as the number of Residual Groups in RCAN.
    Setting an element of the list to True will signal the addition of meta-layers in the corresponding residual block.
    q_layers_inner_residual: Number of q_layers to add within each residual block.
    Set to None to add q_layers to all inner residual blocks.
    Otherwise, other parameters controlling the network can be set akin to normal RCAN.

    Check QRCAN architecture class for further info on internal parameters available.
    """

    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, in_features=3, scheduler=None,
                 scheduler_params=None, style='modulate', perceptual=None, clamp=False, min_mu=-0.2,
                 max_mu=0.8, n_feats=64, srmd_mode=False, **kwargs):
        super(QRCANHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                           **kwargs)

        if 'include_sft_layer' in kwargs and kwargs['include_sft_layer']:  # both sft and srmd modes required tiled metadata
            self.srmd_channel_mode = True
        else:
            self.srmd_channel_mode = srmd_mode

        if srmd_mode:  # for SRMD, the metadata is concatenated directly with the input image
            model_in_features = self.num_metadata + in_features
            self.channel_concat = True
        else:
            model_in_features = in_features

        self.net = QRCAN(scale=scale, in_feats=model_in_features, num_metadata=self.num_metadata,
                         n_feats=n_feats, style=style, **kwargs)

        self.colorspace = 'augmented_rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)

        self.model_name = 'qrcan'
        self.min_mu = min_mu
        self.max_mu = max_mu
        self.base_scaler = np.linspace(0, 1, n_feats)
        self.clamp = clamp
        self.style = style

    @staticmethod
    def gaussian(x, mu, sig=0.2):
        return torch.from_numpy(
            (1 / (np.sqrt(2 * np.pi) * sig)) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))).type(
            torch.float32)

    def scale_qpi(self, qpi):  # TODO: various updates possible here
        scaled_qpi = (qpi * (self.max_mu - self.min_mu)) + self.min_mu
        scalers = []
        for i in range(scaled_qpi.size(0)):
            scalers.append(self.gaussian(self.base_scaler, scaled_qpi[i].squeeze().numpy()))
        full_scalers = torch.stack(scalers)
        if self.clamp:
            full_scalers = torch.clamp(full_scalers, 0, 1)
        return full_scalers.unsqueeze(2).unsqueeze(3)

    def generate_channels(self, x, metadata, keys):  # TODO: create a sub-class containing this function
        if self.srmd_channel_mode:  # need to generate vector metadata not channel-like metadata if using q-injection
            return super().generate_sft_channels(x, metadata, keys)
        else:
            return super().generate_channels(x, metadata, keys)


class QEDSRHandler(QModel):
    """
    Meta-modified EDSR.  Check original EDSR handler/architecture for details on inputs.
    """

    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, in_features=3, num_blocks=16,
                 num_features=64, res_scale=0.1, scheduler=None, scheduler_params=None,
                 perceptual=None, **kwargs):
        super(QEDSRHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                           **kwargs)

        self.net = QEDSR(scale=scale, in_features=in_features, num_features=num_features, num_blocks=num_blocks,
                         res_scale=res_scale, input_para=self.num_metadata, **kwargs)

        self.colorspace = 'augmented_rgb'
        self.im_input = 'unmodified'
        self.activate_device()

        self.model_name = 'qedsr'
        self.criterion = nn.L1Loss()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)


class QSANHandler(QModel):
    """
    Meta-modified SAN.  Check original SAN handler/architecture for details on inputs.
    """

    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, perceptual=None,
                 max_combined_im_size=160000, scheduler=None, scheduler_params=None, **kwargs):
        super(QSANHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                          **kwargs)

        self.net = QSAN(scale=scale, input_para=self.num_metadata)
        self.scale = scale
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)

        self.max_combined_im_size = max_combined_im_size

        self.model_name = 'qsan'

    def forward_chop(self, x, extra_channels, shave=10):
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
                sr_list.append(self.run_chopped_eval(chunk, extra_channels))
        else:
            sr_list = [
                self.forward_chop(patch, extra_channels, shave=shave)
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
        extra_channels = self.generate_channels(x, metadata, metadata_keys).to(self.device)
        if timing:
            tic = time.perf_counter()
        sr_image = self.forward_chop(x, extra_channels)
        if timing:
            toc = time.perf_counter()

        if request_loss:
            return sr_image, self.criterion(sr_image, y), toc - tic if timing else None
        else:
            return sr_image, None, toc - tic if timing else None

    def run_chopped_eval(self, x, extra_channels):
        return super().run_eval(x, y=None, request_loss=False, extra_channels=extra_channels)[0]


class QHANHandler(QModel):
    """
    Meta-modified HAN.  Check original HAN handler/architecture for details on inputs.
    """

    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, perceptual=None,
                 scheduler=None, scheduler_params=None, **kwargs):
        super(QHANHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                          **kwargs)

        self.net = QHAN(scale=scale, num_metadata=self.num_metadata, **kwargs)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)

        self.model_name = 'qhan'


class QRealESRGANHandler(QModel, BaseBSRGANModel):
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
        Main handler for meta-version of the RealESRGAN - very similar to BSRGAN with some small changes.
        Changes include:
        - the naming of model parameters
        - the type of GAN loss
        """
        super(QRealESRGANHandler, self).__init__(device=device,
                                                 model_save_dir=model_save_dir,
                                                 eval_mode=eval_mode,
                                                 **kwargs)

        self.net = QRRDBNet(scale=scale, num_metadata=self.num_metadata, **kwargs)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()

        self.lambda_adv = lambda_adv
        self.lambda_pixel = lambda_pixel
        self.lambda_vgg = lambda_vgg

        self.vgg_feature_layers = vgg_feature_layers
        self.vgg_layer_weights = vgg_layer_weights

        self.model_name = 'qrealesrgan'

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
        self.discriminator.train()

        input_data, extra_channels = self.channel_concat_logic(x, extra_channels, metadata, metadata_keys)
        input_data, extra_channels, y = input_data.to(device=self.device), extra_channels.to(
            device=self.device), y.to(device=self.device)

        if self.curr_epoch < self.pretrain_epochs:  # L1 pre-training
            out = self.run_model(input_data, extra_channels)  # run data through model
            loss = self.criterion(out, y)  # compute L1 loss
            self.pre_train_update(loss)

            loss_G = loss
            loss_content = torch.tensor(0.0, device=self.device)
            loss_GAN = torch.tensor(0.0, device=self.device)
            loss_real = torch.tensor(0.0, device=self.device)
            loss_fake = torch.tensor(0.0, device=self.device)
        else:
            out = self.run_model(input_data, extra_channels)  # run data through model
            loss_G, loss_content, loss_GAN, loss = self.generator_update(out, y)
            loss_real, loss_fake = self.discriminator_update(out, y)

        loss_package = {}

        for _loss, name in zip((loss_G, loss, loss_GAN, loss_content, loss_real, loss_fake),
                               ('train-loss', 'l1-loss', 'gan-loss', 'vgg-loss', 'd-loss-real', 'd-loss-fake')):
            loss_package[name] = _loss.cpu().data.numpy()

        return loss_package, out.detach().cpu()

    def run_model(self, x, extra_channels=None, *args, **kwargs):
        return self.net.forward(x, extra_channels)


class QELANHandler(QModel):
    """
    Main handler for ELAN.
    """

    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, in_features=3, perceptual=None,
                 scheduler=None, scheduler_params=None, **kwargs):
        super(QELANHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                           **kwargs)
        self.net = QELAN(scale=scale, colors=in_features, num_metadata=self.num_metadata, **kwargs)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device)
        self.model_name = 'qelan'
        if scheduler == 'multi_step_lr':
            self.end_epoch_scheduler = True
        else:
            self.end_epoch_scheduler = False

    def run_train(self, x, y, metadata, extra_channels=None, metadata_keys=None, *args, **kwargs):
        self.net.train()
        input_data, extra_channels = self.channel_concat_logic(x, extra_channels, metadata, metadata_keys)
        input_data, extra_channels, y = input_data.to(device=self.device), extra_channels.to(
            device=self.device), y.to(device=self.device)

        out = self.run_model(input_data, extra_channels)
        loss = self.criterion(out, y)
        self.standard_update(loss, scheduler_skip=self.end_epoch_scheduler)

        return loss.detach().cpu().numpy(), out.detach().cpu()

    def run_model(self, x, extra_channels=None, *args, **kwargs):
        return self.net.forward(x, extra_channels)

    def epoch_end_calls(self):
        if self.learning_rate_scheduler is not None and isinstance(self.learning_rate_scheduler,
                                                                   torch.optim.lr_scheduler.MultiStepLR):
            self.learning_rate_scheduler.step()


class MetaBedHandler(QModel):
    """
    Miniature EDSR.  Check Metabed architecture file for further info on parameters.
    """

    def __init__(self, device, model_save_dir, eval_mode=False, lr=1e-4, scale=4, in_features=3, num_blocks=8,
                 num_features=64, res_scale=0.1, scheduler=None, scheduler_params=None, meta_block=None,
                 perceptual=None,
                 use_encoder=None, encoder_pretrain_epochs=None, encoder_loss_scaling=5,
                 freeze_encoder_after_pretrain=None,
                 freeze_decoder_after_pretrain=True, freeze_net_during_pretrain=None, vgg_type='vggface',
                 vgg_mode='p_loss', **kwargs):
        super(MetaBedHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                             **kwargs)

        if meta_block == 'SFT' or in_features > 3:
            self.sft_block = True
        else:
            self.sft_block = False

        if in_features > 3:
            self.channel_concat = True

        self.net = Metabed(scale=scale, in_features=in_features, num_features=num_features, num_blocks=num_blocks,
                           meta_block=meta_block, res_scale=res_scale, input_para=self.num_metadata,
                           use_encoder=use_encoder, **kwargs)

        if meta_block is None:
            self.no_metadata = True
        else:
            self.no_metadata = False

        self.colorspace = 'augmented_rgb'
        self.im_input = 'unmodified'
        self.activate_device()

        self.model_name = 'metabed'
        self.encoder_pretrain_epochs = encoder_pretrain_epochs
        self.use_encoder = use_encoder
        self.encoder_loss_scaling = encoder_loss_scaling

        self.freeze_encoder_after_pretrain = freeze_encoder_after_pretrain
        self.freeze_decoder_after_pretrain = freeze_decoder_after_pretrain

        self.freeze_net_during_pretrain = freeze_net_during_pretrain

        self.criterion = nn.L1Loss()
        self.pretrain_criterion = nn.L1Loss()
        self.training_setup(lr, scheduler, scheduler_params, perceptual, device, vgg_type=vgg_type, vgg_mode=vgg_mode)

    def run_train(self, x, y, metadata, extra_channels=None, metadata_keys=None, *args, **kwargs):
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')

        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)

        input_data, extra_channels = self.channel_concat_logic(x, extra_channels, metadata, metadata_keys)

        if self.no_metadata == False:
            extra_channels = extra_channels.to(device=self.device)

        input_data, y = input_data.to(device=self.device), y.to(device=self.device)

        mult_AE = 0

        if self.use_encoder:
            encoded_metadata = self.net.meta_enc(extra_channels)
            encoded_metadata = encoded_metadata.to(device=self.device)

            decoded_metadata = self.net.meta_dec(encoded_metadata)
            decoded_metadata = decoded_metadata.to(device=self.device)

            loss_AE = self.pretrain_criterion(decoded_metadata, extra_channels)

            if self.encoder_pretrain_epochs and (self.curr_epoch < self.encoder_pretrain_epochs):
                mult_AE = self.encoder_loss_scaling

                # FREEZE METABED
                if self.freeze_net_during_pretrain:
                    for param in self.net.head.parameters():
                        param.requires_grad = False

                    for param in self.net.body.parameters():
                        param.requires_grad = False

                    for param in self.net.final_body.parameters():
                        param.requires_grad = False

                    for param in self.net.final_body.parameters():
                        param.requires_grad = False

            elif self.encoder_pretrain_epochs and (self.curr_epoch >= self.encoder_pretrain_epochs):
                # FREEZE DECODER
                if self.freeze_decoder_after_pretrain:
                    for param in self.net.meta_dec.parameters():
                        param.requires_grad = False

                # FREEZE ENCODER
                if self.freeze_encoder_after_pretrain:
                    for param in self.net.meta_enc.parameters():
                        param.requires_grad = False

                # UNFREEZE METABED
                if self.freeze_net_during_pretrain:
                    for param in self.net.head.parameters():
                        param.requires_grad = True

                    for param in self.net.body.parameters():
                        param.requires_grad = True

                    for param in self.net.final_body.parameters():
                        param.requires_grad = True

                    for param in self.net.final_body.parameters():
                        param.requires_grad = True

                mult_AE = 0
        else:
            loss_AE = torch.tensor(0)

        out = self.run_model(input_data, extra_channels)
        out = out.to(device=self.device)

        loss_SR = self.criterion(out, y)

        scaled_loss_AE = mult_AE * loss_AE
        loss = loss_SR + scaled_loss_AE

        # Standard optimizer step
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss
        if self.grad_clip is not None:  # gradient clipping
            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()  # update network parameters

        if self.learning_rate_scheduler is not None:
            self.learning_rate_scheduler.step()

        if self.use_encoder:
            # Copied from DASR
            loss_package = {}
            for _loss, name in zip((loss, loss_SR, loss_AE, scaled_loss_AE),
                                   ('train-loss', 'l1-loss', 'l1-loss-ae', 'scaled-l1-loss-ae')):
                loss_package[name] = _loss.detach().cpu().numpy()

            return loss_package, out.detach().cpu()
        else:
            return loss.detach().cpu().numpy(), out.detach().cpu()

    def run_model(self, x, extra_channels=None, *args, **kwargs):
        if self.use_encoder:
            extra_channels = self.net.meta_enc(extra_channels)

        return super().run_model(x, extra_channels=extra_channels, *args, **kwargs)

    def run_eval(self, x, y=None, request_loss=False, metadata=None, metadata_keys=None,
                 extra_channels=None, *args, **kwargs):
        input_data, extra_channels = self.channel_concat_logic(x, extra_channels, metadata, metadata_keys)

        return super().run_eval(input_data, y, request_loss=request_loss, extra_channels=extra_channels, **kwargs)

    def generate_channels(self, x, metadata, keys):  # TODO: create a sub-class containing this function
        if self.sft_block:  # need to generate vector metadata not channel-like metadata if using q-injection
            return super().generate_sft_channels(x, metadata, keys)
        else:
            return super().generate_channels(x, metadata, keys)


class MetabedESRGANHandler(MetaBedHandler, ESRGANHandler):
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
                 in_features=3, num_blocks=8, num_features=64,
                 res_scale=0.1, meta_block=None, perceptual=None,
                 use_encoder=None, encoder_pretrain_epochs=None, encoder_loss_scaling=5,
                 freeze_encoder_after_pretrain=None,
                 freeze_decoder_after_pretrain=True, freeze_net_during_pretrain=None,
                 vgg_type='vgg', vgg_mode='p_loss',
                 **kwargs):
        """
        Main handler for MetabedESRGAN. This handler copies a lot of attributes from the Metabed handler and the ESRGAN.
        Please note eval loss is always L1loss, and does not consider extra types of losses.
        """
        super(MetabedESRGANHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                                   **kwargs)

        if meta_block == 'SFT' or in_features > 3:
            self.sft_block = True
        else:
            self.sft_block = False

        if in_features > 3:
            self.channel_concat = True

        self.net = Metabed(scale=scale, in_features=in_features, num_features=num_features, num_blocks=num_blocks,
                           meta_block=meta_block, res_scale=res_scale, input_para=self.num_metadata, **kwargs)

        if meta_block is None:
            self.no_metadata = True
        else:
            self.no_metadata = False

        self.colorspace = 'augmented_rgb'
        self.im_input = 'unmodified'
        self.activate_device()

        self.model_name = 'metabed'
        self.encoder_pretrain_epochs = encoder_pretrain_epochs
        self.use_encoder = use_encoder
        self.encoder_loss_scaling = encoder_loss_scaling

        self.freeze_encoder_after_pretrain = freeze_encoder_after_pretrain
        self.freeze_decoder_after_pretrain = freeze_decoder_after_pretrain

        self.freeze_net_during_pretrain = freeze_net_during_pretrain

        self.lambda_adv = lambda_adv
        self.lambda_pixel = lambda_pixel
        self.pretrain_epochs = pretrain_epochs

        # Specific Optimizer, Discriminator and Scheduler Config (only for training)
        self.optimizer = {}
        self.learning_rate_scheduler = {}

        if not self.eval_mode:
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
            self.vgg_extractor = perceptual_loss_mechanism(vgg_type, mode=vgg_mode, device=device)
            self.vgg_extractor.to(self.device)
            self.vgg_extractor.eval()

            # additional error criteria
            self.criterion = nn.L1Loss()
            self.pretrain_criterion = nn.L1Loss()
            self.criterion_GAN = nn.BCEWithLogitsLoss()
            self.criterion_content = nn.L1Loss()

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
        self.discriminator.train()

        # Get the metadata for the metabed model
        input_data, extra_channels = self.channel_concat_logic(x, extra_channels, metadata, metadata_keys)

        if self.no_metadata == False:
            extra_channels = extra_channels.to(device=self.device)

        input_data, y = input_data.to(device=self.device), y.to(device=self.device)

        mult_AE = 0

        # Add the encoder with the specified training/freezing
        if self.use_encoder:
            encoded_metadata = self.net.meta_enc(extra_channels)
            encoded_metadata = encoded_metadata.to(device=self.device)

            decoded_metadata = self.net.meta_dec(encoded_metadata)
            decoded_metadata = decoded_metadata.to(device=self.device)

            loss_AE = self.pretrain_criterion(decoded_metadata, extra_channels)

            if self.encoder_pretrain_epochs and (self.curr_epoch < self.encoder_pretrain_epochs):
                mult_AE = self.encoder_loss_scaling

                # FREEZE METABED
                if self.freeze_net_during_pretrain:
                    for param in self.net.head.parameters():
                        param.requires_grad = False

                    for param in self.net.body.parameters():
                        param.requires_grad = False

                    for param in self.net.final_body.parameters():
                        param.requires_grad = False

                    for param in self.net.final_body.parameters():
                        param.requires_grad = False

            elif self.encoder_pretrain_epochs and (self.curr_epoch >= self.encoder_pretrain_epochs):
                # FREEZE DECODER
                if self.freeze_decoder_after_pretrain:
                    for param in self.net.meta_dec.parameters():
                        param.requires_grad = False

                # FREEZE ENCODER
                if self.freeze_encoder_after_pretrain:
                    for param in self.net.meta_enc.parameters():
                        param.requires_grad = False

                # UNFREEZE METABED
                if self.freeze_net_during_pretrain:
                    for param in self.net.head.parameters():
                        param.requires_grad = True

                    for param in self.net.body.parameters():
                        param.requires_grad = True

                    for param in self.net.final_body.parameters():
                        param.requires_grad = True

                    for param in self.net.final_body.parameters():
                        param.requires_grad = True

                mult_AE = 0
        else:
            loss_AE = torch.tensor(0)

        # Run data through model
        out = self.run_model(input_data, extra_channels)
        out = out.to(device=self.device)

        # Compute L1 loss of model
        loss_SR = self.criterion(out, y)

        # Scale the loss of the AE
        scaled_loss_AE = mult_AE * loss_AE

        # Combine SR loss and AE loss
        # If AE is not used it will be set to 0
        loss = loss_SR + scaled_loss_AE

        if self.curr_epoch < self.pretrain_epochs:  # L1 pre-training
            loss_package = self.pre_train_update(loss)
            return loss_package, out.detach().cpu()
        else:
            loss_G, loss_content, loss_GAN, _ = self.generator_update(out, y)
            loss_D = self.discriminator_update(out, y)

        loss_package = {}

        for _loss, name in zip((loss_G, loss, loss_GAN, loss_content, loss_D),
                               ('train-loss', 'l1-loss', 'gan-loss', 'vgg-loss', 'discriminator-loss')):
            loss_package[name] = _loss.cpu().data.numpy()

        return loss_package, out.detach().cpu()
