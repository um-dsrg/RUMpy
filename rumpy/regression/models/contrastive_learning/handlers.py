import torch

from .supmoco import SupMoCo
from .weak_con import WeakCon
from .moco import MoCo
from . import BaseContrastive
from .base_models import AdjustedStandardModel

from rumpy.sr_tools.loss_functions import SupConLoss


class MocoContrastiveHandler(BaseContrastive):
    def __init__(self, device, model_save_dir, eval_mode=False, output_size=10,
                 scheduler=None,
                 scheduler_params=None,
                 lr=1e-4,
                 model_name=None,
                 crop_count=2,
                 moco_t=0.07,  # 0.2 to be the same as MoCoV2
                 **kwargs):
        super(MocoContrastiveHandler, self).__init__(device=device,
                                                     model_save_dir=model_save_dir,
                                                     eval_mode=eval_mode,
                                                     **kwargs)

        self.crop_count = crop_count

        self.net = MoCo(base_encoder=self.define_encoder_model(model_name), T=moco_t, positives=crop_count-1)
        self.activate_device()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.training_setup(lr, scheduler, scheduler_params, device=device, perceptual=None)

    def run_model(self, x, *args, **kwargs):
        embedding, q = self.net.forward(x, x, get_q=True, **kwargs)
        return embedding, q

    def run_train(self, x, y, tag=None, mask=None, *args, **kwargs):

        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)

        if self.crop_count == 2:
            x = x.to(device=self.device)

            _, output, target = self.net(im_q=x[:, 0:3, ...], im_k=x[:, 3:, ...])
        else:
            x = x.view(-1, 3, x.shape[2], x.shape[3]).to(device=self.device)

            batch_count = int(x.shape[0]/self.crop_count)
            indices = [i * self.crop_count for i in range(batch_count)]
            non_indices = [i for i in range(x.shape[0]) if i not in indices]

            _, output, target = self.net(im_q=x[indices], im_k=x[non_indices])

        target = target.to(device=self.device)
        loss_contrast = self.criterion(output, target)
        self.standard_update(loss_contrast)

        return loss_contrast.detach().cpu().numpy(), output.detach().cpu()


class SupMoCoHandler(BaseContrastive):
    def __init__(self, device, model_save_dir, eval_mode=False, output_size=10,
                 scheduler=None,
                 scheduler_params=None,
                 lr=1e-4,
                 model_name='default',
                 crop_count=2,
                 moco_t=0.07,
                 data_type='noise',
                 dropdown=None,
                 dropdown_metadata_target=None,
                 include_direct_loss=False,
                 direct_loss_only=False,
                 contrastive_dropdown=True,
                 **kwargs):
        super(SupMoCoHandler, self).__init__(device=device,
                                             model_save_dir=model_save_dir,
                                             eval_mode=eval_mode,
                                             **kwargs)
        self.crop_count = crop_count
        self.temperature = moco_t

        self.data_type = data_type

        # with dropdown mode on, an additional MLP is added to the standard encoder,
        # which brings down the vector output to the specified size.
        # With include_direct_loss, a direct dropdown vs target metadata loss (L1) can be computed.
        if dropdown is not None and contrastive_dropdown:
            vector_dim = dropdown
        else:
            vector_dim = 256

        self.net = SupMoCo(base_encoder=self.define_encoder_model(model_name),
                           positives_per_class=crop_count - 1, dim=vector_dim,
                           contrastive_dropdown=contrastive_dropdown,
                           T=moco_t, device=device, dropdown=dropdown)
        self.activate_device()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.training_setup(lr, scheduler, scheduler_params, device=device, perceptual=None)

        if include_direct_loss and dropdown is None:
            raise RuntimeError('Dropdown needs to be enabled to use direct loss during training.')

        self.include_direct_loss = include_direct_loss
        self.dropdown = dropdown
        self.dropdown_metadata_target = dropdown_metadata_target
        self.contrastive_dropdown = contrastive_dropdown
        self.direct_loss_only = direct_loss_only
        self.target_loss = torch.nn.L1Loss()

    def run_train(self, x, y, tag=None, mask=None, *args, **kwargs):

        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)
        x = x.view(-1, 3, x.shape[2], x.shape[3]).to(device=self.device)

        labels = self.class_logic(y, kwargs['metadata_keys'])

        batch_count = int(x.shape[0]/self.crop_count)
        indices = [i * self.crop_count for i in range(batch_count)]
        non_indices = [i for i in range(x.shape[0]) if i not in indices]

        embedding, logits, full_labels, q = self.net(x[indices], x[non_indices], labels.squeeze())
        full_labels = full_labels.to(self.device)

        loss_contrast = self.criterion(logits, full_labels)  # apply cross-entropy loss (final part of supmoco loss)
        if self.include_direct_loss:  # extracts metadata from y, then calculates L1 using dropdown q versus target metadata
            m_keys = kwargs['metadata_keys']
            mask = [True if key[0] in self.dropdown_metadata_target else False for key in m_keys]
            selected_target = torch.ones(y.size(0), self.dropdown).to(device=self.device)

            for index, _ in enumerate(selected_target):
                added_info = y[index][mask].to(device=self.device)
                selected_target[index, ...] = selected_target[index, :] * added_info

            loss_regression = self.target_loss(selected_target, q['dropdown_q'])
            if self.direct_loss_only:
                loss = loss_regression
            else:
                loss = loss_regression + loss_contrast
        else:
            loss = loss_contrast

        self.standard_update(loss)

        if self.include_direct_loss:
            loss_package = {}
            for _loss, name in zip((loss, loss_regression, loss_contrast), ('train-loss', 'regression-loss', 'contrastive-loss')):
                loss_package[name] = _loss.cpu().data.numpy()
        else:
            loss_package = loss_contrast.detach().cpu().numpy()

        return loss_package, embedding.detach().cpu()

    def run_model(self, x, *args, **kwargs):
        embedding, q = self.net.forward(x, x, get_q=True, **kwargs)
        return embedding, q


class WeakConHandler(BaseContrastive):
    def __init__(self, device, model_save_dir, eval_mode=False, output_size=10,
                 scheduler=None,
                 scheduler_params=None,
                 lr=1e-4,
                 model_name='default',
                 crop_count=2,
                 moco_t=0.07,
                 data_type='noise',
                 **kwargs):
        super(WeakConHandler, self).__init__(device=device,
                                             model_save_dir=model_save_dir,
                                             eval_mode=eval_mode,
                                             **kwargs)
        self.crop_count = crop_count
        self.temperature = moco_t

        self.data_type = data_type

        self.net = WeakCon(base_encoder=self.define_encoder_model(model_name),
                           positives_per_class=crop_count - 1,
                           T=moco_t, device=device)
        self.activate_device()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.training_setup(lr, scheduler, scheduler_params, device=device, perceptual=None)

    def run_train(self, x, y, tag=None, mask=None, *args, **kwargs):
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)
        x = x.view(-1, 3, x.shape[2], x.shape[3]).to(device=self.device)

        vectors = self.vector_logic(y, kwargs['metadata_keys']).to(device=self.device)

        batch_count = int(x.shape[0] / self.crop_count)
        indices = [i * self.crop_count for i in range(batch_count)]
        non_indices = [i for i in range(x.shape[0]) if i not in indices]

        embedding, logits, full_labels = self.net(x[indices], x[non_indices], vectors.squeeze())
        full_labels = full_labels.to(self.device)

        loss_contrast = self.criterion(logits, full_labels)  # apply cross-entropy loss (final part of supmoco loss)

        self.standard_update(loss_contrast)

        return loss_contrast.detach().cpu().numpy(), embedding.detach().cpu()

    def run_model(self, x, *args, **kwargs):
        embedding, q = self.net.forward(x, x, get_q=True, **kwargs)
        return embedding, q


class SupConHandler(BaseContrastive):
    def __init__(self, device, model_save_dir, eval_mode=False, output_size=10,
                 scheduler=None,
                 model_name='default',
                 scheduler_params=None,
                 lr=1e-4,
                 crop_count=2,
                 **kwargs):
        super(SupConHandler, self).__init__(device=device,
                                            model_save_dir=model_save_dir,
                                            eval_mode=eval_mode,
                                            **kwargs)
        net_class = self.define_encoder_model(model_name)
        if net_class.__name__ != 'Encoder' and net_class.__name__ != 'IDMN':
            self.net = AdjustedStandardModel(net_class(num_classes=256))  # TODO: remove hard-coding
        else:
            self.net = net_class()
        self.activate_device()
        self.criterion = SupConLoss()
        self.training_setup(lr, scheduler, scheduler_params, device=device, perceptual=None)
        self.crop_count = crop_count

    def run_train(self, x, y, tag=None, mask=None, *args, **kwargs):

        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)
        x = x.view(-1, 3, x.size()[2], x.size()[3]).to(device=self.device)

        embedding, q = self.net(x)

        labels = self.class_logic(y, kwargs['metadata_keys'])

        loss_contrast = self.criterion(q.view(-1, self.crop_count, q.size()[1]), labels)

        self.standard_update(loss_contrast)

        return loss_contrast.detach().cpu().numpy(), embedding.detach().cpu()

    def run_model(self, x, *args, **kwargs):
        embedding, q = self.net.forward(x)
        return embedding, q

