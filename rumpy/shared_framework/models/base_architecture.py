import math
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn, optim as optim

from rumpy.shared_framework.models import define_model
from rumpy.sr_tools.loss_functions import PerceptualMechanism
from rumpy.sr_tools.stats import load_statistics
from rumpy.sr_tools.helper_functions import standard_metric_epoch_selection
from rumpy.shared_framework.configuration.gpu_check import mps_check


class BaseModel(nn.Module):
    """
    Basic model architecture template.  All models can inherit and expand from this base formula.
    Functionality provided:
    basic optimizer/scheduler setup, GPU usage, basic training/eval, model saving/loading, model diagnostics.
    Models should be called from ModelInterface, and not directly using the base architectures.
    """
    def __init__(self, device, model_save_dir, eval_mode, grad_clip=None, loss_masking=False, **kwargs):
        """
        :param device: GPU device ID (or 'cpu').
        :param model_save_dir: Model save directory.
        :param eval_mode: Set to true to turn off training functionality.
        :param grad_clip: If gradient clipping is required during training, set gradient limit here.
        :param loss_masking: Set to true to activate loss masking mechanism.
        """
        super(BaseModel, self).__init__()

        if device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = device

        # default values for parameters specified in model handlers
        self.criterion = nn.L1Loss()
        self.optimizer = None
        self.net = None
        self.face_finder = False
        self.im_input = None
        self.colorspace = None
        self.steps = None
        self.eval_request_loss = True  # can be modified in regression handlers
        self.loss_masking = loss_masking

        if grad_clip == 0:
            self.grad_clip = None
        else:
            self.grad_clip = grad_clip
        self.model_save_dir = model_save_dir
        self.eval_mode = eval_mode
        self.curr_epoch = 0
        self.state = {}
        self.learning_rate_scheduler = None
        self.legacy_load = True  # loading system which ensures weight names match as expected

        # by default uses class name for model identifier
        self.model_name = self.__class__.__name__.split('Handler')[0].lower()

    def activate_device(self):
        """
        Sends model to specified GPU.
        """
        self.net.to(self.device)

    def set_multi_gpu(self, device_ids=None):
        """
        Spreads model throughout multiple GPUs.
        :param device_ids: GPUs available for model sharing.
        """
        self.net = nn.DataParallel(self.net, device_ids=device_ids)
        if len(self.net.device_ids) > 1:
            print('Model sent to multiple GPUs:', ', '.join([str(d_id) for d_id in self.net.device_ids]))

    def define_optimizer(self, optim_weights, lr=1e-4, optimizer_params=None, optimizer_type='Adam'):
        """
        Defines an optimizer and attaches it to the specified trainable weights.
        :param optim_weights: Trainable weights.
        :param lr: Learning rate.
        :param optimizer_params: Optimizer specific parameters (dict).
        :param optimizer_type: Optimizer type (adam or rmsprop)
        :return: Optimizer object.
        """
        if optimizer_type.lower() == 'adam':
            if optimizer_params is not None:
                beta_1 = optimizer_params['beta_1']
                beta_2 = optimizer_params['beta_2']
                betas = (beta_1, beta_2)
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, optim_weights), lr=lr, betas=betas)
            else:
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, optim_weights), lr=lr)
        elif optimizer_type.lower() == 'rmsprop':
            if optimizer_params is not None:
                alpha = optimizer_params['alpha']
                optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, optim_weights), lr=lr, alpha=alpha)
            else:
                optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, optim_weights), lr=lr)

        return optimizer

    def define_scheduler(self, base_optimizer, scheduler, scheduler_params):
        """
        Defines a learning rate scheduler and attaches it to a specified optimizer.
        :param base_optimizer: Optimizer to manage.
        :param scheduler: Scheduler name.
        :param scheduler_params: Specific scheduler parameters.
        :return: Instantiated scheduler.
        """
        if scheduler == 'cosine_annealing_warm_restarts':
            learning_rate_scheduler = \
                optim.lr_scheduler.CosineAnnealingWarmRestarts(base_optimizer, T_mult=scheduler_params['t_mult'],
                                                               T_0=scheduler_params['restart_period'],
                                                               eta_min=scheduler_params['lr_min'])

        elif scheduler == 'one_cycle_lr':
            learning_rate_scheduler = optim.lr_scheduler.OneCycleLR(base_optimizer, max_lr=scheduler_params['lr_max'],
                                                                    total_steps=scheduler_params['total_steps'],  # epochs * steps per epoch
                                                                    anneal_strategy=scheduler_params['anneal_strategy'])

        elif scheduler == 'multi_step_lr':
            learning_rate_scheduler = optim.lr_scheduler.MultiStepLR(base_optimizer,
                                                                     milestones=scheduler_params['milestones'],
                                                                     gamma=scheduler_params['gamma'])
        elif scheduler == 'custom_dasr':
            def dasr_no_encoder_scheduler(epoch):
                if epoch < 225:
                    lr = 1e-4
                else:
                    cycle = (epoch - 100) // 125
                    lr = 1e-4 * math.pow(0.5, cycle)
                return lr

            def dasr_scheduler(epoch):
                if epoch < 60:
                    lr = 1e-3
                elif epoch < 225:
                    lr = 1e-4
                else:
                    cycle = (epoch - 100) // 125
                    lr = 1e-4 * math.pow(0.5, cycle)
                return lr

            def dasr_short_scheduler(epoch):
                if epoch < 21:  # 60/600 = 0.1, 0.1 * 210 = 21
                    lr = 1e-3
                elif epoch < 79:  # 225/600 = 0.375, 0.375 * 210 = 78.75
                    lr = 1e-4
                else:
                    cycle = (epoch - 35) // 44  # 100 -> 35, 125 -> 43.75
                    lr = 1e-4 * math.pow(0.5, cycle)
                return lr

            if scheduler_params['train_type'] == 'long':
                lambda_scheduler = dasr_scheduler
            elif scheduler_params['train_type'] == 'short':
                lambda_scheduler = dasr_short_scheduler
            elif scheduler_params['train_type'] == 'no_encoder_long':
                lambda_scheduler = dasr_no_encoder_scheduler
            else:
                raise RuntimeError('Need to select from long or short scheduler type for DASR.')

            learning_rate_scheduler = optim.lr_scheduler.LambdaLR(base_optimizer, lr_lambda=lambda_scheduler)

        elif scheduler == 'custom_contrastive':
            def contrastive_scheduler(batch_iter):
                if batch_iter < 260:
                    lr = 0.1
                else:
                    lr = 5e-4
                return lr

            learning_rate_scheduler = optim.lr_scheduler.LambdaLR(base_optimizer, lr_lambda=contrastive_scheduler)

        elif scheduler == 'step_lr':
            learning_rate_scheduler = optim.lr_scheduler.StepLR(base_optimizer,
                                                                step_size=scheduler_params['step_size'],
                                                                gamma=scheduler_params['gamma'])
        elif scheduler == 'custom':
            learning_rate_scheduler = optim.lr_scheduler.LambdaLR(base_optimizer, lr_lambda=scheduler_params['function'])
        else:
            raise RuntimeError('%s scheduler not implemented' % scheduler)
        return learning_rate_scheduler

    def training_setup(self, lr, scheduler, scheduler_params, perceptual, device, optimizer_params=None, vgg_type='vgg', vgg_mode='p_loss'):

        if not self.eval_mode:
            self.optimizer = self.define_optimizer(self.net.parameters(), lr=lr, optimizer_params=optimizer_params)
            if scheduler is not None:
                self.learning_rate_scheduler = self.define_scheduler(base_optimizer=self.optimizer,
                                                                     scheduler=scheduler,
                                                                     scheduler_params=scheduler_params)

        if perceptual is not None and self.eval_mode is False:
            self.criterion = PerceptualMechanism(lambda_per=perceptual, device=device, vgg_type=vgg_type, vgg_mode=vgg_mode)

    @staticmethod
    def extract_model_parameters(model):
        """
        Returns model parameters (without additional abstraction)
        """
        if isinstance(model, nn.DataParallel):
            return model.module.state_dict()
        else:
            return model.state_dict()

    def _extract_multiple_models_from_dict(self, model_store, model_key, config='save', load_state=None):
        """
        Saves or loads sub-models from a single dictionary/variable location.
        :param model_store: Base model dictionary, or single model variable
        :param model_key: Model group name
        :param config: 'load' or 'save'
        :param load_state: Specify load location if loading
        :return: Model store is modified in-place.
        """
        if type(model_store) == dict:
            for name, sub_model in model_store.items():
                if config == 'load':
                    model_store[name].load_state_dict(load_state[name])
                elif config == 'save':
                    self.state[name] = self.extract_model_parameters(sub_model)
        else:
            if config == 'load':
                model_store.load_state_dict(load_state[model_key])
            elif config == 'save':
                self.state[model_key] = self.extract_model_parameters(model_store)

    def save_model(self, model_save_name, extract_state_only=False, minimal=False):
        """
        Saves current model and other salient parameters to file.
        :param model_save_name: model save name (not the same as experiment save name).
        :param extract_state_only: Set to true to only extract the save state of the model,
        rather than actually save it to file.
        :param minimal: Set to true to only save necessary model components for downstream eval.
        :return: None (model saved directly to file)
        """
        net_params = self.extract_model_parameters(self.net)  # removes model abstraction

        self.state['network'] = net_params  # save network parameters and other variables

        self.state['model_name'] = self.model_name
        self.state['model_epoch'] = self.curr_epoch

        if not minimal:
            # saving optimizer, or multiple optimizers if available
            self._extract_multiple_models_from_dict(self.optimizer, 'optimizer', config='save')

            # saving scheduler, or multiple schedulers if available
            if self.learning_rate_scheduler is not None:
                self._extract_multiple_models_from_dict(self.learning_rate_scheduler, 'scheduler_G', config='save')

            # saving discriminator, or multiple discriminators if available
            if hasattr(self, 'discriminator'):
                self._extract_multiple_models_from_dict(self.discriminator, 'discriminator', config='save')

            if hasattr(self, 'steps'):
                self.state['steps'] = self.steps

        if extract_state_only:
            return self.state

        torch.save(self.state, f=os.path.join(self.model_save_dir, "{}_{}".format(model_save_name, self.curr_epoch)))

    def load_setup(self, load_override, model_save_name, model_idx):
        """
        Prepares generic device and filepath info for model loading.
        """
        if self.device == torch.device('cpu'):  # device location to map to, when pre-loading
            loc = self.device
        elif isinstance(self.device, int) or self.device.isnumeric():
            loc = "cuda:%s" % self.device
        elif mps_check(self.device):  # currently a bug with loading CUDA models into MPS
            loc = torch.device('cpu')
        else:
            raise RuntimeError('Device %s not recognized' % self.device)

        if load_override is None:
            load_file = os.path.join(self.model_save_dir, "{}_{}".format(model_save_name, str(model_idx)))
        else:
            load_file = os.path.join(load_override, "{}_{}".format(model_save_name, str(model_idx)))
        return load_file, loc

    def load_model(self, model_save_name, model_idx, legacy=False, load_override=None, preloaded_state=None, config_changes=None, skip_scheduler_load=False, skip_optimizer_load=False):
        """
        Loads selected model and other parameters from specified location
        :param model_save_name: saved model name.
        :param model_idx: model epoch number.
        :param legacy:  Set to True if model saved with legacy system.
        :param load_override: Override default model save location for loading.
        :param preloaded_state: State to load in, if this has been pre-loaded.
        :param skip_scheduler_load: Set to true to skip loading the optimizer when loading a saved model.
        :param skip_optimizer_load: Set to true to skip loading the optimizer when loading a saved model.
        :return: state dictionaries.
        """

        load_file, loc = self.load_setup(load_override, model_save_name, model_idx)

        if preloaded_state is None:  # actual model loading step
            state = torch.load(f=load_file, map_location=loc)
        else:
            state = preloaded_state

        if config_changes is not None and 'values_changed' in config_changes:
            if 'root[\'internal_params\'][\'lr\']' in config_changes['values_changed']:
                scheduler_keys = [k for k in state.keys() if 'scheduler' in k.lower()]

                for key in scheduler_keys:
                    state[key]['base_lrs'] = [config_changes['values_changed']['root[\'internal_params\'][\'lr\']']['new_value']]
                    state[key]['_last_lr'] = [config_changes['values_changed']['root[\'internal_params\'][\'lr\']']['new_value']]

                optimizer_keys = [k for k in state.keys() if 'optimizer' in k.lower()]

                for key in optimizer_keys:
                    state[key]['param_groups'][0]['lr'] = config_changes['values_changed']['root[\'internal_params\'][\'lr\']']['new_value']

        if 'dan' in state['model_name']:
            # pre-trained DAN models sometimes don't provide init kernel.
            # If this is the case, the defaults are applied here instead.
            self.dan_check(state)

        optimizer_load = not skip_optimizer_load

        if self.model_name == 'supmoco':
            # catering for the case where dropdown MLP not available in saved weights
            if hasattr(self.net.encoder_q, 'drop_mlp') and 'encoder_q.drop_mlp.0.weight' not in state['network']:
                optimizer_load = False  # optimizer loading needs to be skipped - parameters will not match correctly
                params = self.extract_model_parameters(self.net)
                for key, val in params.items():
                    if 'drop_mlp' in key:
                        state['network'][key] = val

            # if there is a mismatch in queue shape, then this has to be reset
            if self.net.queue.shape[0] != state['network']['queue'].shape[0]:
                state['network']['queue'] = self.net.queue
                state['network'].pop('queue_labels', None)
                state['network']['queue_ptr'] = self.net.queue_ptr

            elif 'queue_labels' in state['network']:  # exception for backwards compatibility
                if not hasattr(self.net, 'queue_labels'):
                    # sometimes, the network won't have the queue pre-instantiated.
                    # This step creates a random queue here before loading in the new queue.
                    if hasattr(state['network'], 'num_classes'):  # currently no models save the class count.
                        num_classes = state['network'].num_classes
                    else:
                        num_classes = 0
                    self.net.register_classes(num_classes)

        elif 'queue_labels' in state['network']:  # removes queue if supmoco pre-trained model loaded and not running supmoco
            state['network'].pop('queue_labels', None)

        if self.model_name == 'weakcon':  # exception for backwards compatibility
            if not hasattr(self.net, 'queue_vectors') and 'queue_vectors' in state['network']:
                # sometimes, the network won't have the queue pre-instantiated.
                # This step creates a random queue here before loading in the new queue.
                vector_len = state['network']['queue_vectors'].size()[0]
                self.net.register_vector(vector_len)

        if legacy:  # legacy compatibility fix
            self.net.load_state_dict(state_dict=self.legacy_switch(state['network'], qrealesrgan_fix=True if 'qrealesrgan' in state['model_name'] else False))
        else:
            self.net.load_state_dict(state_dict=state['network'])

        if mps_check(self.device):
            self.net.to(device=self.device)  # sends network to device after model loaded

        if not self.eval_mode:
            # loading optimizer, or multiple optimizers if available
            if optimizer_load:
                self._extract_multiple_models_from_dict(self.optimizer, 'optimizer', config='load', load_state=state)

            # loading scheduler, or multiple schedulers if available
            if not skip_scheduler_load:
                if self.learning_rate_scheduler is not None:
                    self._extract_multiple_models_from_dict(self.learning_rate_scheduler, 'scheduler_G', config='load',
                                                            load_state=state)

            # loading discriminator, or multiple discriminators if available
            if hasattr(self, 'discriminator'):
                self._extract_multiple_models_from_dict(self.discriminator, 'discriminator', config='load',
                                                        load_state=state)

            if hasattr(self, 'steps'):
                self.steps = state['steps']

        self.set_epoch(state['model_epoch'])

        if state['model_name'] == 'qpircan':  # legacy conversion system
            state['model_name'] = 'qrcan'

        print('Loaded model uses the following architecture:', state['model_name'])
        return state

    @staticmethod
    def legacy_switch(state_dict, qrealesrgan_fix=False):
        """
        Method that retains some backwards-compatibility with older models when loading state dicts.
        Simply removes some extra prefixes in param strings.
        """
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:13] == 'model.module.':
                new_state_dict[k[13:]] = v
            elif k[:6] == 'model.':
                new_state_dict[k[6:]] = v
            elif qrealesrgan_fix and 'q_block.attribute_integrator' in k:
                continue
            else:
                new_state_dict[k] = v
        return new_state_dict

    def dan_check(self, state_dict):
        """
        Simple check to convert pre-trained official DAN model to local implementation.
        :param state_dict: State of model to be loaded.
        :return: None, dict updated in-place
        """
        if 'init_kernel' not in state_dict['network'] and hasattr(self.net, 'init_kernel'):
            state_dict['network']['init_kernel'] = self.net.init_kernel
        if 'init_ker_map' not in state_dict['network']:
            state_dict['network']['init_ker_map'] = self.net.init_ker_map

    def standard_update(self, loss, scheduler_skip=False):
        """
        Standard network backprop update.
        :param loss: Loss value (connected to network gradient graph)
        :param scheduler_skip: Set to true to skip scheduler call
        """
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss

        if self.grad_clip is not None:  # gradient clipping
            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)

        self.optimizer.step()  # update network parameters

        if self.learning_rate_scheduler is not None and not scheduler_skip:  # update learning rate
            self.learning_rate_scheduler.step()

    def run_model(self, x, *args, **kwargs):
        """
        Main model call.  Can be overridden by child models.
        """
        return self.net.forward(x)

    def find_loss(self, out, y):  # TODO: find a generic way to calculate and separate component losses (both train and eval)
        return self.criterion(out, y)  # compute loss

    def get_binary_masks(self, masks):  # TODO: if mask could be used in N,C,H,W format, things would be more straightforward...
        new_masks = torch.zeros_like(masks)
        non_black_pixels = (masks.permute((0, 2, 3, 1)) != torch.tensor((0, 0, 0))).all(-1)
        new_masks[non_black_pixels.unsqueeze(1).expand(-1, 3, -1, -1)] = 1
        return new_masks

    def run_train(self, x, y, tag=None, mask=None, keep_on_device=False, scheduler_skip=False, *args, **kwargs):
        """
        Basic training iteration.  Can be overridden in child models.
        :param x: Input variable.
        :param y: Output ground-truth.
        :param tag: Image names.
        :param mask: Loss mask to apply on output/HR image
        :param keep_on_device: Return output, but do not move to CPU.
        :param scheduler_skip: Set to true to skip scheduler call.
        :param args: Additional arguments used by child models.
        :param kwargs: Additional arguments used by child models.
        :return: Loss value (detached) and network output.
        """
        if self.eval_mode:
            raise RuntimeError('Model initialized in eval mode, training not possible.')
        self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)
        x, y = x.to(device=self.device), y.to(device=self.device)
        out = self.run_model(x, image_names=tag, **kwargs)  # run data through model
        if self.loss_masking:
            binary_mask = self.get_binary_masks(mask).to(device=self.device)
            out *= binary_mask
            y *= binary_mask
        loss = self.find_loss(out, y)
        self.standard_update(loss, scheduler_skip=scheduler_skip)  # takes care of optimizer calls, backprop and scheduler calls

        if keep_on_device:
            return loss.detach().cpu().numpy(), out.detach()
        else:
            return loss.detach().cpu().numpy(), out.detach().cpu()

    # TODO: probably best to convert this into a system where eval outputs extra items (e.g. loss and timing) in a separate dict
    def run_eval(self, x, y=None, request_loss=False, tag=None, timing=False, keep_on_device=False, *args, **kwargs):
        """
        Runs a model evaluation for the given data batch.
        :param x: input (full-channel).
        :param y: target (full-channel).
        :param request_loss: Set to true to also compute network loss with current criterion.
        :param tag: Image names.
        :param timing: Set to true to time network run-time.
        :param keep_on_device: Set this to true to keep final output on input device (GPU).
        Otherwise, result will always be transferred to CPU.
        :return: calculated output, loss and timing.
        """
        self.net.eval()  # sets the system to validation mode

        with torch.no_grad():
            x = x.to(device=self.device)
            if timing:
                tic = time.perf_counter()
            out = self.run_model(x, image_names=tag, **kwargs)  # forward the data in the model
            if timing:
                toc = time.perf_counter()
            if request_loss and y is not None:
                y = y.to(device=self.device)
                loss = self.find_loss(out, y).detach().cpu().numpy()  # compute loss
            else:
                loss = None

        if isinstance(out, tuple):  # only required for contrastive models - TODO can this be removed?
            return out, loss, toc - tic if timing else None
        if keep_on_device:
            return out.detach(), loss, toc - tic if timing else None
        else:
            return out.detach().cpu(), loss, toc - tic if timing else None

    def run_forensic(self, x, *args, **kwargs):
        """
        Runs an eval iteration, while also requesting further internal model state information.
        """
        self.net.eval()
        with torch.no_grad():
            x = x.to(device=self.device)
            out, data = self.net.forensic(x, **kwargs)
        return out.cpu().detach(), data

    def print_parameters_model_list(self, model_list, model_names_list):
        """
        Prints the number of parameters for a list of given models.

        :param model_list: List of model objects.
        :param model_names_list: List of model names.
        """
        print('----------------------------')
        print('Per-model Parameters')
        for model, model_name in zip(model_list, model_names_list):
            model_total_params = sum(p.numel() for p in model.parameters())
            print(model_name + ':', model_total_params)

    def print_parameters(self, verbose=False):
        """
        Reports how many trainable parameters are available in the model, and where they are distributed.
        :return: Total trainable parameters.
        """
        if verbose:
            print('----------------------------')
            print('Parameter names:')
        total_num_parameters = 0
        for name, value in self.named_parameters():
            if verbose:
                print(name, value.shape)
            total_num_parameters += np.prod(value.shape)
        if verbose:
            print('Total number of trainable parameters:', total_num_parameters)
            print('----------------------------')
        return total_num_parameters

    def print_status(self):
        """
        Prints out any additional diagnostics.
        """
        raise NotImplementedError

    def epoch_end_calls(self):
        """
        Any end of epoch function calls should be implemented here.
        """
        pass

    def extra_diagnostics(self):
        """
        Empty method for models to print out any extra details on first instantiation, if required.
        """
        pass

    def pre_training_model_load(self):
        # TODO: can this be integrated with warm-start system?
        """
        Use this method to pre-load models trained from other experiments, if required.
        """
        pass

    def verify_eval(self):
        """
        Use this method to signal that the eval routine should not be run.
        e.g. to signal eval routine not to run if model is still pre-training.
        """
        return True

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    @staticmethod
    def best_model_selection_criteria(log_dir=None, log_file='summary.csv', model_metadata=None,
                                      stats=None, stats_dir=None, base_metric='val-PSNR'):
        """
        Use this method to signal any non-standard method for selecting the best epoch from a training run.
        Basic implementation always assumes the best epoch can be chosen by evaluating the highest metric value.
        :return: Best model epoch number
        """
        if stats_dir and stats is None:
            stats = load_statistics(log_dir, log_file, config='pd')  # loads model training stats if not provided

        return standard_metric_epoch_selection(base_metric, stats)


class MultiModel(BaseModel):
    """
    Model container that houses and manages multiple individual models.
    """
    def __init__(self, multi_params=None, **kwargs):
        super(MultiModel, self).__init__(**kwargs)
        self.child_models = {}
        for key, model in multi_params.items():  # child models prepared here
            self.child_models[key] = define_model(**kwargs, **model)

    def set_multi_gpu(self, device_ids=None):
        """
        Runs through all child models, and sends them to the specified GPU/s.
        """
        for model in self.child_models.keys():
            self.child_models[model].net = nn.DataParallel(self.child_models[model].net, device_ids=device_ids)
        devices = list(self.child_models.values())[0].net.device_ids
        if len(devices) > 1:
            print('Model sent to multiple GPUs:', ', '.join([str(d_id) for d_id in devices]))

    def define_scheduler(self, **kwargs):
        raise NotImplementedError('Scheduler/Optimizer are handled individually for each model.')

    def define_optimizer(self, **kwargs):
        raise NotImplementedError('Scheduler/Optimizer are handled individually for each model.')

    def save_model(self, model_save_name, extract_state_only=False, minimal=False):
        states = {'global_epoch': self.curr_epoch}
        for model in self.child_models.keys():
            states[model] = self.child_models[model].save_model(model_save_name, extract_state_only=True,
                                                                minimal=minimal)
        if extract_state_only:
            return states
        else:
            torch.save(states, f=os.path.join(self.model_save_dir, "{}_{}".format(model_save_name, str(self.curr_epoch))))

    def load_model(self, model_save_name, model_idx, legacy=True, load_override=None, **kwargs):

        load_file, loc = self.load_setup(load_override, model_save_name, model_idx)

        multi_state = torch.load(f=load_file, map_location=loc)
        self.curr_epoch = multi_state['global_epoch']  # global epoch ID, internal models might
        # have different epoch counts

        for model in self.child_models.keys():
            self.child_models[model].load_model(model_save_name, model_idx, legacy, load_override,
                                                preloaded_state=multi_state[model])

    def run_train(self, **kwargs):
        raise NotImplementedError('No base MultiModel training available '
                                  '(need to implement specific routine for each model).')

    def run_eval(self, **kwargs):
        raise NotImplementedError('No base MultiModel eval available '
                                  '(need to implement specific routine for each model).')

    def print_parameters(self, verbose=False):
        total_num_parameters = 0
        for key, model in self.child_models.items():
            if verbose:
                print('--------')
                print('Parameters for model %s:' % key)
            total_num_parameters += model.print_parameters(verbose=verbose)
        return total_num_parameters

    def epoch_end_calls(self):
        for model in self.child_models.values():
            model.epoch_end_calls()

    def extra_diagnostics(self):
        print('This is a multi-model system.')
        for idx, model in enumerate(self.child_models.values()):
            print('Model %d has the %s architecture (with %d parameters), and was loaded at local epoch %d.' % (
                idx, model.model_name, model.print_parameters(), model.curr_epoch))

    def set_epoch(self, epoch):
        self.curr_epoch = epoch
        for model in self.child_models.values():
            model.set_epoch(epoch)

    def get_learning_rate(self):
        lrs = {}
        for key, model in self.child_models.items():
            lrs['%s_learning_rate' % key] = model.get_learning_rate()
        return lrs
