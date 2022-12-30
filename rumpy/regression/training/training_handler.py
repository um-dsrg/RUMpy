import itertools
import os
import time
from collections import defaultdict
import shutil

import numpy as np
import pandas as pd
import tqdm
from prefetch_generator import BackgroundGenerator


from rumpy.regression.models.interface import RegressionInterface
from rumpy.shared_framework.training.base_handler import BaseTrainingHandler
from rumpy.sr_tools.helper_functions import create_dir_if_empty
from rumpy.regression.evaluation.eval_hub import ContrastiveEval
import rumpy.shared_framework.configuration.constants as sconst


class RegressionTrainingHandler(BaseTrainingHandler):
    def __init__(self,
                 # all
                 npz_eval_save=False, run_eval=False,
                 eval_frequency=1, save_model_outputs=False,
                 # contrastive
                 run_tsne=True,
                 run_umap=False,
                 cont_tsne_perplexity=40.0,
                 cont_umap_n_neighbors=25,
                 cont_umap_metric='cosine',
                 cont_umap_min_dist=0.5,
                 cont_mapping_normalize=True,
                 cont_save_plots=True,
                 cont_plot_extension='png',
                 cont_plot_dpi=100,
                 warm_start_model=None,
                 labelling_strategy='default',
                 *args, **kwargs):
        """
        :param npz_eval_save: Set to true to save model output to .npz format.
        :param run_eval: Set to true to enable eval.
        :param eval_frequency: Eval epoch running frequency.
        :param save_model_outputs: Set to true to also save output embedding/q values for each eval run.
        :param warm_start_model: Select from list of available standard pretrained models to continue training from.

        Contrastive-only params:
        :param cont_tsne_perplexity: Perplexity value to use for TSNE.
        :param cont_mapping_normalize: Set to true to normalize TSNE outputs.
        :param cont_save_plots: Set to false to disable plot saving during eval.
        :param cont_plot_extension: Extension for plots produced by eval system.
        :param cont_plot_dpi: Plot dpi.

        :param labelling_strategy: Set the labelling for the metadata system so that it can be used for analysis.

        :param args: All other standard training arguments.
        :param kwargs: All other standard training arguments.
        """

        if warm_start_model is not None:
            self.warm_start_setup(warm_start_model, kwargs['save_loc'], kwargs['experiment_name'])
        super(RegressionTrainingHandler, self).__init__(*args, **kwargs)
        self.npz_eval_safe = npz_eval_save
        self.eval_frequency = eval_frequency
        self.run_eval = run_eval
        self.save_output = save_model_outputs

        if self.model.model.regressor_type == 'contrastive':

            self.run_tsne = run_tsne
            self.run_umap = run_umap
            self.save_plots = cont_save_plots
            self.contrastive_eval_hub = ContrastiveEval()
            self.contrastive_eval_hub.define_embedding_length(self.model.model.get_embedding_len())
            self.contrastive_eval_hub.register_metadata(self.val_data.dataset, labelling_strategy)
            # TODO: find way to feed hyperparams to contrastive eval handler
            self.contrastive_eval_hub.config_output_plots(file_extension=cont_plot_extension, dpi=cont_plot_dpi)
            tsne_config = {
                'perplexity': cont_tsne_perplexity,
                'normalize_fit': cont_mapping_normalize
            }
            umap_config = {
                'n_neighbors': cont_umap_n_neighbors,
                'metric': cont_umap_metric,
                'min_dist': cont_umap_min_dist,
                'normalize_fit': cont_mapping_normalize
            }
            self.combined_config = {**tsne_config, **umap_config}

    def setup_model(self, **kwargs):
        return RegressionInterface(**kwargs)

    def warm_start_setup(self, pretrained_model, model_folder, new_experiment):
        pretrained_model_folder = os.path.join(sconst.code_base_directory, 'regression', 'pretrained_networks',
                                               pretrained_model)
        if os.path.isdir(pretrained_model_folder):

            copyfiles = ['pretrained_config.toml', os.path.join('result_outputs', 'summary.csv')]

            available_models = os.listdir(os.path.join(pretrained_model_folder, 'saved_models'))

            for model in available_models:
                if 'train_model' in model:
                    copyfiles.append(os.path.join('saved_models', model))

            full_input_files = [os.path.join(pretrained_model_folder, p) for p in copyfiles]
            full_output_files = [os.path.join(model_folder, new_experiment, p) for p in copyfiles]

            if os.path.isfile(full_output_files[0]):
                print('Pretrained config already in place.')
            else:
                new_exp = os.path.join(model_folder, new_experiment)
                create_dir_if_empty(new_exp, os.path.join(new_exp, 'result_outputs'), os.path.join(new_exp, 'saved_models'))

                for inp, outp in zip(full_input_files, full_output_files):
                    shutil.copy2(inp, outp)
                print('Pretrained config and checkpoints transferred to model folder.')
        else:
            raise RuntimeError('The warm start model selected is not available.')

    def contrastive_eval(self, epoch_idx):

        output_package = defaultdict(list)

        if len(self.contrastive_eval_hub.metadata_hyperparams) == 0:
            self.contrastive_eval_hub.register_hyperparams(self.val_data.dataset.lr_base)

        if self.model.model.model_name == 'supmoco' and self.model.model.dropdown is not None:
            dropdown = True
            dropdown_size = self.model.model.dropdown
        else:
            dropdown = False
            dropdown_size = 1
        # TODO: allow user to select between tsne and umap for training eval
        image_names = self.contrastive_eval_hub.generate_data_encoding(run_tsne=self.run_tsne,
                                                                       run_umap=self.run_umap,
                                                                       data_loader=self.val_data,
                                                                       has_dropdown=dropdown,
                                                                       dropdown_size=dropdown_size,
                                                                       model=self.model,
                                                                       **self.combined_config)
        self.contrastive_eval_hub.initialize_output_folder(self.model.logs, 'epoch_%d_results' % epoch_idx)

        if self.save_plots:
            # TSNE
            if self.run_tsne:

                if 'gaussian_noise_scale' in self.contrastive_eval_hub.metadata_keys:
                    self.contrastive_eval_hub.plot_noise(plot_magnitudes=True, rep_type='tsne')

                if 'jpeg_quality_factor' in self.contrastive_eval_hub.metadata_keys:
                    self.contrastive_eval_hub.plot_compression(rep_type='tsne')

                if 'jpeg_quality_factor' in self.contrastive_eval_hub.metadata_keys and 'gaussian_noise_scale' in self.contrastive_eval_hub.metadata_keys:
                    self.contrastive_eval_hub.plot_combined_noise_compression(rep_type='tsne')

                if 'kernel_type' in self.contrastive_eval_hub.metadata_keys:
                    self.contrastive_eval_hub.plot_blur(rep_type='tsne')

            # UMAP
            if self.run_umap:
                if 'gaussian_noise_scale' in self.contrastive_eval_hub.metadata_keys:
                    self.contrastive_eval_hub.plot_noise(plot_magnitudes=True, rep_type='umap')

                if 'jpeg_quality_factor' in self.contrastive_eval_hub.metadata_keys:
                    self.contrastive_eval_hub.plot_compression(rep_type='umap')
                    self.contrastive_eval_hub.plot_combined_noise_compression(rep_type='umap')

                if 'kernel_type' in self.contrastive_eval_hub.metadata_keys:
                    self.contrastive_eval_hub.plot_blur(rep_type='umap')

        if self.save_output:
            output_package['embedding'] = self.contrastive_eval_hub.data_encodings.tolist()
            output_package['q'] = self.contrastive_eval_hub.data_q.tolist()
            output_package['image_name'] = list(itertools.chain.from_iterable(image_names))

        return defaultdict(list), output_package, self.contrastive_eval_hub.base_folder

    def regression_eval(self, epoch_idx):

        current_epoch_losses = defaultdict(list)
        output_package = defaultdict(list)

        with tqdm.tqdm(total=len(self.val_data)) as pbar_val:
            prefetcher = BackgroundGenerator(self.val_data)
            for _, batch in enumerate(prefetcher):
                out_vector, loss, _ = self.model.net_run_and_process(**batch, request_loss=True)

                output_package['output'].append(list(out_vector.cpu().numpy().flatten()))

                output_package['image_name'].append(batch['tag'])

                current_epoch_losses["val-loss"].append(loss)
                diag_string = 'loss: {:.4f}, '.format(loss)

                # displays diagnostics
                pbar_val.update(1)
                pbar_val.set_description(diag_string[:-2])

        output_package['image_name'] = list(itertools.chain.from_iterable(output_package['image_name']))

        output_folder = os.path.join(self.model.logs, 'epoch_%d_results' % epoch_idx)
        if self.save_output:
            create_dir_if_empty(output_folder)

        return current_epoch_losses, output_package, output_folder

    def eval(self, epoch_idx):
        """
        This function takes care of a single eval run - including metric calculation and logging.
        :param epoch_idx: Current epoch number.
        :return: Full epoch metrics (dict).
        """
        if epoch_idx % self.eval_frequency == 0 and self.run_eval:
            model_run_start = time.perf_counter()

            if self.model.model.regressor_type == 'contrastive':
                current_epoch_losses, output_package, output_folder = self.contrastive_eval(epoch_idx)
            elif self.model.model.regressor_type == 'standard':
                current_epoch_losses, output_package, output_folder = self.regression_eval(epoch_idx)
            else:
                raise RuntimeError('Unrecognised model type.')

            if self.save_output:
                full_results = pd.DataFrame.from_dict(output_package).set_index(['image_name'])

                # saving results for each image to csv
                if self.npz_eval_safe:
                    np.savez(os.path.join(output_folder, 'output_results.npz'), all_kernels=full_results)
                else:
                    full_results.to_csv(os.path.join(output_folder, 'output_results.csv'))

            model_run_end = time.perf_counter()

            print('Evaluation time: %.4f seconds' % (model_run_end - model_run_start))

            return current_epoch_losses
        else:
            return defaultdict(list)
