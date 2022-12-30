from rumpy.shared_framework.models.base_architecture import BaseModel
from rumpy.SISR.models.generative_face_models.discriminators import *
from rumpy.SISR.models.generative_face_models.generators import *


class FaceGANHandler(BaseModel):
    def __init__(self, device, model_save_dir, eval_mode=False, scale=8,
                 main_lr=2e-4, discriminator_lr=2e-4,
                 main_optimizer_params={'beta_1': 0.5, 'beta_2': 0.999},
                 discriminator_optimizer_params={'beta_1': 0.5, 'beta_2': 0.999},
                 main_scheduler=None,
                 main_scheduler_params=None,
                 latent_dim=100,
                 use_scheduler=False,
                 **kwargs):
        """
        Based on the tutorial in the 'Machine Learning Mastery' book.
        """
        super(FaceGANHandler, self).__init__(device=device,
                                             model_save_dir=model_save_dir,
                                             eval_mode=eval_mode,
                                             **kwargs)

        self.net = GANGenerator(latent_dim=latent_dim)

        self.colorspace = 'rgb'
        self.im_input = 'unmodified'
        self.activate_device()

        self.latent_dim = latent_dim
        self.use_scheduler = use_scheduler

        # Specific Optimizer, Discriminator and Scheduler Config (only for training)
        self.optimizer = {}
        self.learning_rate_scheduler = {}

        if not self.eval_mode:
            self.optimizer['main_optimizer'] = self.define_optimizer(self.net.parameters(),
                                                                     lr=main_lr, optimizer_params=main_optimizer_params)

            if main_scheduler is not None:
                self.learning_rate_scheduler['main_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['main_optimizer'],
                    scheduler=main_scheduler,
                    scheduler_params=main_scheduler_params)

            # discriminator model
            self.discriminator = GANDiscriminator()
            self.discriminator.to(self.device)
            self.optimizer['discrim_optimizer'] = self.define_optimizer(self.discriminator.parameters(),
                                                                        lr=discriminator_lr,
                                                                        optimizer_params=discriminator_optimizer_params)
            if main_scheduler is not None:  # same scheduler used for discriminator as for main generator
                self.learning_rate_scheduler['discrim_scheduler'] = self.define_scheduler(
                    base_optimizer=self.optimizer['discrim_optimizer'],
                    scheduler=main_scheduler,
                    scheduler_params=main_scheduler_params)

            # additional error criteria
            self.criterion_GAN = nn.BCELoss()

    def discriminator_update(self, gen_image, ref_image):
        for p in self.discriminator.parameters():  # TODO: is this required?
            p.requires_grad = True

        self.optimizer['discrim_optimizer'].zero_grad()

        # Real predictions and targets
        real_predictions = self.discriminator(ref_image)
        real_targets = real_predictions.new_ones(real_predictions.size()) * 1.0

        loss_D_real = self.criterion_GAN(real_predictions, real_targets)
        loss_D_real.backward()

        # Fake predictions and targets
        fake_predictions = self.discriminator(gen_image.detach())
        fake_targets = fake_predictions.new_ones(fake_predictions.size()) * 0.0

        loss_D_fake = self.criterion_GAN(fake_predictions, fake_targets)
        loss_D_fake.backward()

        self.optimizer['discrim_optimizer'].step()

        if self.use_scheduler:
            self.learning_rate_scheduler['discrim_scheduler'].step()

        real_pred_labels = real_predictions > 0.5
        real_accuracy = torch.mean((real_pred_labels == real_targets).float())

        fake_pred_labels = fake_predictions > 0.5
        fake_accuracy = torch.mean((fake_pred_labels == fake_targets).float())

        return loss_D_real, loss_D_fake, real_accuracy, fake_accuracy

    def generator_update(self, gen_image):
        for p in self.discriminator.parameters():  # TODO: is this really needed?
            p.requires_grad = False

        self.optimizer['main_optimizer'].zero_grad()

        gen_predictions = self.discriminator(gen_image)
        gen_targets = gen_predictions.new_ones(gen_predictions.size()) * 1.0

        loss_G = self.criterion_GAN(gen_predictions, gen_targets)
        loss_G.backward()

        self.optimizer['main_optimizer'].step()

        if self.use_scheduler:
            self.learning_rate_scheduler['main_scheduler'].step()

        return loss_G

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

        # Load the real image (x is not needed since we only need the HR images)
        y = y.to(device=self.device)

        # Choose half of the batch of the real images at random
        batch_size = y.size(dim=0)
        half_batch_size = batch_size // 2

        indices = torch.randperm(batch_size)[:half_batch_size]
        y_half_batch = y[indices]
        y_half_batch.to(device=self.device)

        # Scale real images to be between -1 and 1, to match with the generator
        y_half_batch = (y_half_batch * 2) - 1

        # Get the randomly generated latent dimensions for the discriminator
        fake_samples_d = torch.rand(self.latent_dim * half_batch_size, device=self.device)
        fake_samples_d_half_batch = torch.reshape(fake_samples_d, (half_batch_size, -1))

        # Get the randomly generated latent dimensions for the generator
        fake_samples_g = torch.rand(self.latent_dim * batch_size, device=self.device)
        fake_samples_g_r = torch.reshape(fake_samples_g, (batch_size, -1))

        # Update the discriminator with fake and real images
        gen_images_d = self.net.forward(fake_samples_d_half_batch)
        loss_D_real, loss_D_fake, acc_D_real, acc_D_fake = self.discriminator_update(gen_images_d, y_half_batch)

        # Update the generator with real images through the discriminator loss
        gen_images_g = self.net.forward(fake_samples_g_r)
        loss_G = self.generator_update(gen_images_g)

        # Rescale images because the generator outputs images between -1 and 1 because of tanh
        gen_images_g = (gen_images_g - 1.0) / 2.0

        loss_package = {}

        for _loss, name in zip((loss_G, loss_D_real, loss_D_fake, acc_D_real, acc_D_fake),
                               ('train-loss', 'd-loss-real', 'd-loss-fake', 'd-acc-real', 'd-acc-fake')):
            loss_package[name] = _loss.cpu().data.numpy()

        return loss_package, gen_images_g.detach().cpu()

    def extra_diagnostics(self):
        if not self.eval_mode:
            models = [self.net, self.discriminator]
            model_names = ['Generator', 'Discriminator']
            self.print_parameters_model_list(models, model_names)

    def get_learning_rate(self):  # TODO: this could also be generalised for all multi discriminator models
        lrs = {}
        for key, optimizer in self.optimizer.items():
            lrs['%s_learning_rate' % key] = optimizer.param_groups[0]['lr']
        return lrs

    def verify_eval(self):
        # for now, evaluation does not have any additional metrics to offer
        return False
