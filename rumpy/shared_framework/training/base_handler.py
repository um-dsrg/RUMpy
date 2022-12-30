import math
import os
import random
import time
from collections import defaultdict
from datetime import date, datetime
import importlib
from colorama import init, Fore
import numpy as np
import torch
import tqdm
from prefetch_generator import BackgroundGenerator

from rumpy.shared_framework.configuration import constants as sconst
from rumpy.shared_framework.training.data_setup import sisr_data_setup
from rumpy.sr_tools.helper_functions import replace_char_in_recursive_dict, create_dir_if_empty, clean_models
from rumpy.sr_tools.metrics import Metrics
from rumpy.sr_tools.stats import plot_stats, save_statistics
from rumpy.sr_tools.visualization import safe_image_save

aim_spec = importlib.util.find_spec("aim")  # only imports aim if this is available
if aim_spec is not None:
    from aim import Run

init()  # colorama setup


class BaseTrainingHandler:
    def __init__(self,
                 # general params
                 experiment_name='experiment-%s' % date.today().strftime("%b-%d-%Y"), save_loc=sconst.results_directory,
                 aim_track=False, aim_home=os.path.join(sconst.results_directory, 'SISR'),
                 # model params
                 model_params=None, gpu='off', sp_gpu=1, best_load_metric='val-PSNR',
                 # data params
                 data_params=None,
                 # train params
                 num_epochs=None, continue_from_epoch=None, max_im_val=1.0, metrics=None, seed=8,
                 model_cleanup_frequency=None, cleanup_metric='val-PSNR', epoch_cutoff=None, run_lpips_on_gpu=False,
                 vgg_gallery=None, id_source=None, early_stopping_metric='val-PSNR', early_stopping_patience=None,
                 overwrite_data=False, branch_root=None, new_branch=False, new_branch_name=None, logging='visual',
                 full_loss_logging=False, save_samples=True, new_params_override_load=None, skip_scheduler_load=False, skip_optimizer_load=False,
                 eval_frequency=1, **kwargs):
        """
        Initializes a super-res ML experiment training handler
        :param experiment_name: model save name (defaults to date/time if not provided)
        :param aim_track: Set to True to track diagnostics using Aim
        :param aim_home: Home directory for aim tracking
        :param save_loc: model save location
        :param model_params: model instantiation parameters and sp
        :param gpu: 'single' - use one gpu, 'multi' - use all available gpus or 'off' - use CPU only
        :param sp_gpu: Select which specific gpu to be used
        :param best_load_metric: Metric to use for selecting best/last epoch
        :param data_params: parameters for setting up data handler
        :param num_epochs: number of epochs to train for
        :param continue_from_epoch: Restart training from a particular save point
        :param max_im_val: image excepted max pixel value
        :param metrics: metrics to monitor throughout training
        :param seed: Random generator seed initialization
        :param model_cleanup_frequency: Number of epochs to wait before wiping unneeded models
        :param cleanup_metric: Metric to use when cleaning up models (for selecting the best model)
        :param epoch_cutoff: Epoch cutoff point (also considering any epochs previously run)
        :param run_lpips_on_gpu: Set to true to run LPIPS metrics using GPU
        :param vgg_gallery: vgg reference gallery to monitor face recognition performance
        :param id_source: id file for assigning id to each image
        :param early_stopping_metric: Metric to track when triggering early stopping
        :param early_stopping_patience: number of epochs after which training will end if no progress continues to be made
        :param overwrite_data: set to true and handler will overwrite any saved models with new data
        :param branch_root: Name of branch to use for training
        :param new_branch: Set to true to construct a new model branch
        :param new_branch_name: Name of new branch, if required
        :param logging: Type of logging to perform during experiment - set to 'visual' to print out loss plts
        :param save_samples:  Set to true to save image samples after each training epoch
        :param full_loss_logging: Set to true to output all the loss components during training, set to false to show only train-loss
        :param new_params_override_load: Set to true to force model config of new loaded model to match that of the new params specified.
        :param skip_scheduler_load: Set to true to skip loading the scheduler when loading a saved model.
        :param skip_optimizer_load: Set to true to skip loading the optimizer when loading a saved model.
        :param eval_frequency: The frequency of running the evaluation in number of epochs. A higher values means less frequent evaluation
        :param kwargs: Any runoff parameters
        """

        # essential experimental setup
        self.experiment_name = experiment_name
        self.num_epochs = num_epochs
        self.logging = logging
        self.full_loss_logging = full_loss_logging
        self.save_samples = save_samples
        self.stop_patience = early_stopping_patience
        self.early_stop_metric = early_stopping_metric
        self.overwrite = overwrite_data
        self.model_cleanup_frequency = model_cleanup_frequency
        self.cleanup_metric = cleanup_metric
        self.aim_track = aim_track

        self.eval_frequency = eval_frequency
        self.metrics_list = metrics

        # random seed initialization
        torch.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Implementation setup
        self.best_val_model_idx = 0
        self.best_val_model_early_metric = 0
        self.model_name = model_params['name']  # model architecture name
        self.max_im_val = max_im_val  # image maximum pixel value (affects PSNR calculation)
        self.new_branch = new_branch

        if 'multi_frame_config' in data_params:
            loss_masking = data_params['multi_frame_config']['use_masks']
        else:
            loss_masking = False

        # sets up model, loads in any provided checkpoint and initializes experiment directory
        self.model = self.setup_model(model_loc=save_loc, experiment=experiment_name, gpu=gpu, sp_gpu=sp_gpu,
                                      mode='train', new_params=model_params, load_epoch=continue_from_epoch,
                                      save_subdir=branch_root, best_load_metric=best_load_metric,
                                      new_params_override_load=new_params_override_load,
                                      skip_scheduler_load=skip_scheduler_load,
                                      skip_optimizer_load=skip_optimizer_load,
                                      loss_masking=loss_masking)

        # new branch setup logic
        if new_branch and continue_from_epoch:  # only setup a new branch if start point is defined
            if not new_branch_name:  # use default branch name if not provided
                new_branch_name = 'branch_epoch_%d' % continue_from_epoch
        elif not new_branch and continue_from_epoch:
            # Forces a new branch if the selected epoch is not the last epoch available (to preserve results)
            last_epoch = self.model.stats['epoch'].max()
            if continue_from_epoch != last_epoch and continue_from_epoch != 'last':
                self.new_branch = True
                new_branch_name = 'branch_epoch_%d' % continue_from_epoch
                print('%sNew branch created as model not training from last epoch.%s' % (Fore.RED, Fore.RESET))

        if self.new_branch:
            self.model.init_new_branch(new_branch_name)  # prepares new folders for branch

        # epoch logic and run naming
        self.starting_epoch = self.model.model_epoch  # extracts kick-off epoch
        if continue_from_epoch is None:  # sets a unique run name
            run_name = experiment_name + '_%s' % datetime.today().strftime("%Hh-%Mm-%Ss-%b-%d-%Y")
        else:
            run_name = 'continuation_from_epoch_%d_' % self.model.model_epoch + experiment_name + \
                       '_%s' % datetime.today().strftime("%Hh-%Mm-%Ss-%b-%d-%Y")
            self.starting_epoch += 1  # start training from next epoch after loaded epoch
        self.run_name = run_name

        if epoch_cutoff is not None:
            self.num_epochs = epoch_cutoff - self.starting_epoch  # set cutoff based on previously run epochs
            print('Epoch count set to %d' % self.num_epochs)

        # prepares provided training and eval datasets
        self.train_data, self.val_data = sisr_data_setup(scale=model_params['internal_params']['scale'],
                                                         **model_params,
                                                         **self.model.configuration,
                                                         qpi_sort=False, **data_params)

        # Metric Setup
        if metrics is not None:
            self.metric_hub = Metrics(metrics, id_source=id_source, vgg_gallery=vgg_gallery,
                                      lpips_device=sp_gpu if run_lpips_on_gpu else torch.device('cpu'))
        else:
            self.metric_hub = None

        # set up and configure an aim tracker - all logs saved to .aim folder in Aim home directory
        if aim_track:
            store_params = {
                'model_parameters': model_params,
                'data_parameters': data_params,
                'train_parameters': {'num_epochs': num_epochs,
                                     'continue_from_epoch': continue_from_epoch,
                                     'seed': seed,
                                     'epoch_cutoff': epoch_cutoff}
            }
            self.aim_session = self.aim_setup(aim_home, experiment_name, self.run_name, store_params)

    def setup_model(self, **kwargs):
        raise NotImplementedError('Model instantiation function not selected.')

    @staticmethod
    def aim_setup(aim_home, experiment_name, run_name, store_params=None, system_tracking_interval=60):
        """
        Sets up an aim session for tracking metrics.
        :param aim_home: Aim save folder location.  Will create a .aim folder here.
        :param experiment_name: Base experiment name.
        :param run_name: Unique run name.
        :param store_params: Dictionary of parameters to store with run.
        :param system_tracking_interval: Time interval (seconds) used to track system (GPU/CPU etc)
        :return: Aim session object
        """

        if aim_spec is None:  # confirms aim is available
            raise RuntimeError('To activate Aim logging, please install aim using pip install aim')

        aim_session = Run(experiment=experiment_name, repo=aim_home, run_hash=run_name,
                          system_tracking_interval=system_tracking_interval)

        if store_params:
            for key, content in replace_char_in_recursive_dict(store_params).items():
                aim_session[key] = content  # passes over constant parameters to Aim

        return aim_session

    def train(self):
        """
        Function that takes care of a single training epoch -
        model is trained using each input batch, and losses/learning rates are logged.
        :return: Full epoch losses (dict).
        """
        current_epoch_losses = defaultdict(list)

        with tqdm.tqdm(total=len(self.train_data)) as pbar_train:
            prefetcher = BackgroundGenerator(self.train_data)  # data prefetcher improves running time
            data_load_start = time.time()
            for batch in prefetcher:
                data_load_end = time.time()
                losses, _ = self.model.train_batch(**batch)  # entire training scheme occurs here

                if type(losses) is dict:    # takes care to log all salient losses
                    for l_name, l_num in losses.items():
                        current_epoch_losses[l_name].append(l_num)
                    loss = losses['train-loss']
                    if self.full_loss_logging:
                        pbar_description_string = ', '.join(['{}: {:.4f}'.format(l_n, l_i) for l_n, l_i in losses.items()])
                    else:
                        pbar_description_string = 'train-loss: {:.4f}'.format(loss)
                else:
                    loss = losses
                    current_epoch_losses['train-loss'].append(loss)
                    pbar_description_string = 'loss: {:.4f}'.format(loss)

                compute_time_end = time.time()
                compute_time = compute_time_end - data_load_end
                data_time = data_load_end - data_load_start
                pbar_train.update(1)
                pbar_train.set_description(
                    "{}, data load time: {:.4f}s, compute time: {:.4f}s, compute efficiency: {:.2f}%".format(
                        pbar_description_string, data_time, compute_time, 100*(compute_time/(data_time+compute_time))))  # displays current batch metrics
                data_load_start = time.time()

        learning_rates = self.model.get_learning_rate()  # extracts model learning rates for logging purposes
        if type(learning_rates) is dict:
            for m_key, m_lr in learning_rates.items():
                current_epoch_losses[m_key].append(m_lr)
        else:
            current_epoch_losses['learning-rate'].append(learning_rates)

        self.model.epoch_end_calls()  # any model-specific epoch end calls (e.g. for a scheduler)

        return current_epoch_losses

    def eval(self, epoch_idx):
        """
        This function takes care of single eval epoch - including metric calculation and logging.
        :param epoch_idx: Current epoch number.
        :return: Full epoch metrics (dict).
        """
        current_epoch_losses = defaultdict(list)
        with tqdm.tqdm(total=len(self.val_data)) as pbar_val:
            prefetcher = BackgroundGenerator(self.val_data)  # speeds up data loading
            data_load_start = time.time()
            for index, batch in enumerate(prefetcher):
                data_load_end = time.time()
                y, im_names = batch['hr'], batch['tag']
                rgb_out, ycbcr_out, loss, timing = self.model.net_run_and_process(**batch, request_loss=True)

                # ensures that a Y-channel only image is produced for each batch,
                # to allow for standardised metric calculations
                if 'rgb' in self.model.configuration['colorspace']:
                    y_proc = self.model.colorspace_convert(y, colorspace='rgb')
                else:
                    y_proc = self.model._standard_image_formatting(y.numpy())

                # collect and record metrics based on eval run
                current_epoch_losses["val-loss"].append(loss)
                if self.metric_hub is not None:
                    metric_package, _ = self.metric_hub.run_metrics(ycbcr_out, references=y_proc,
                                                                    max_value=self.max_im_val, key='val',
                                                                    probe_names=[im_name.split('.')[0] for im_name
                                                                                 in im_names])
                    for metric, result in metric_package.items():
                        current_epoch_losses[metric].extend(result)

                # saves a single batch sample as a representative result
                if index == 0 and self.save_samples:
                    samples_folder = os.path.join(self.model.logs, 'epoch_%d_samples' % epoch_idx)
                    create_dir_if_empty(samples_folder)
                    im_names = [name.replace(os.sep, '_') for name in im_names]
                    safe_image_save(rgb_out, samples_folder, im_names, config='rgb')

                compute_time_end = time.time()

                pbar_val.update(1)

                compute_time = compute_time_end - data_load_end
                data_time = data_load_end - data_load_start

                diag_string = 'loss: {:.4f}, data load time: {:.2f}s, compute time: {:.2f}s, compute efficiency: {:.2f}%, '.format(
                    loss, data_time, compute_time, 100 * (compute_time / (data_time + compute_time)))

                for metric in metric_package.keys():
                    diag_string += '{}: {:.4f}, '.format(metric, np.mean(metric_package[metric]))
                pbar_val.set_description(diag_string[:-2])
                data_load_start = time.time()

        return current_epoch_losses

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations according to spec,
        saving the model and results after each epoch.
        :return: Complete loss package (dict).
        """
        if self.model.mode == 'eval':
            raise RuntimeError('Model initialized in eval mode, training not possible.')

        total_losses = defaultdict(list)
        if self.starting_epoch != 0:  # reloads old stats from file
            total_losses = defaultdict(list, self.model.stats.to_dict(orient='list'))
            if self.starting_epoch != len(total_losses['epoch']):  # truncates stats if starting epoch is not the last epoch run
                for key in total_losses.keys():
                    total_losses[key] = total_losses[key][0:self.starting_epoch]
            if self.aim_track:  # loads up Aim with previous metrics
                for key, val in total_losses.items():
                    for epoch, item in enumerate(val):
                        self.aim_session.track(item, name=key.replace('-', '_'),
                                               epoch=epoch)

        improvement_count = 0  # used when tracking early stopping

        for i, epoch_idx in enumerate(range(self.starting_epoch, self.starting_epoch+self.num_epochs)):

            # epoch initializations
            epoch_start_time = time.time()
            print('Running epoch', epoch_idx)
            self.model.set_epoch(epoch_idx)

            if i == 0:  # Test to ensure no data will be overwritten with this run
                self.model.save(override=self.overwrite, dry_run=True)

            print('Training Run:')
            training_loss = self.train()

            if not self.model.model.verify_eval() or epoch_idx % self.eval_frequency != 0:
                print('Validation run not possible for this epoch.')
                eval_loss = {}
            else:
                print('Validation Run:')
                eval_loss = self.eval(epoch_idx)

            current_epoch_losses = {**training_loss, **eval_loss}  # combines all metrics

            # Computing statistics
            for key, value in current_epoch_losses.items():
                avg_val = np.nanmean(value)
                # removes nan values - which allows for tokenizing certain keys if tracked only in a few batches
                if math.isnan(avg_val):
                    avg_val = 0
                if key in total_losses:
                    total_losses[key].append(avg_val)  # get mean of all metrics of current epoch
                else:
                    # populates dict with zero values in case a new metric
                    # comes up during training (e.g. when switching model loss type)
                    total_losses[key] = [0] * len(total_losses['epoch'])
                    total_losses[key].append(avg_val)
                if self.aim_track:
                    self.aim_session.track(avg_val.item(), name=key.replace('-', '_'), epoch=epoch_idx)

            # this block makes sure to continue populating stats if they are not produced in this particular epoch
            for key in total_losses.keys():
                if key not in current_epoch_losses and key != 'epoch':
                    total_losses[key].append(0)

            total_losses['epoch'].append(epoch_idx)

            # Saving and reporting statistics
            if self.logging == 'visual':
                if eval_loss:
                    keynames_list = [['train-loss', 'val-loss']]
                else:
                    keynames_list = [['train-loss']]

                if 'PSNR' in self.metrics_list:
                    keynames_list = keynames_list + [['val-PSNR']]

                if 'SSIM' in self.metrics_list:
                    keynames_list = keynames_list + [['val-SSIM']]

                if 'LPIPS' in self.metrics_list:
                    keynames_list = keynames_list + [['val-LPIPS']]

                plot_stats(stats_dict=total_losses, keynames=keynames_list,
                           experiment_log_dir=self.model.logs, filename='loss_plots.pdf')

            # Saves current model checkpoint
            self.model.save(override=self.overwrite)

            # save results to file
            save_statistics(experiment_log_dir=self.model.logs, filename='summary.csv',
                            stats_dict=total_losses,
                            selected_data=epoch_idx if (self.starting_epoch != 0 or i > 0) else None,
                            append=True if (self.starting_epoch != 0 or i > 0) else False)

            out_string = " ".join(["{}_{:.4f}".format(key, np.mean(value))
                                   for key, value in current_epoch_losses.items()])

            # cleans old checkpoints, if set to do so
            if self.model_cleanup_frequency is not None and i != 0 and i % self.model_cleanup_frequency == 0:
                clean_models(self.model.base_folder, clean_samples=False, base_metric=self.cleanup_metric)

            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            if self.aim_track:
                self.aim_session.track(epoch_elapsed_time, name='epoch_time', epoch=epoch_idx)

            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}/{}:".format(epoch_idx, self.starting_epoch + self.num_epochs-1), out_string,
                  "Epoch duration:", epoch_elapsed_time, "seconds")
            print('-------------')

            if 'val-PSNR' in current_epoch_losses:
                val_mean_metric = np.mean(current_epoch_losses[self.early_stop_metric])

                if val_mean_metric > self.best_val_model_early_metric:  # early stopping check
                    self.best_val_model_early_metric = val_mean_metric
                    self.best_val_model_idx = epoch_idx
                    improvement_count = 0
                else:
                    improvement_count += 1

                if improvement_count == self.stop_patience:
                    print('%sStopping model training, validation loss has plateaued.%s' % (Fore.GREEN, Fore.RESET))
                    break

        return total_losses
