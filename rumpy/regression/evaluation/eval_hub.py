from rumpy.sr_tools.data_handler import SuperResImages
from rumpy.regression.models.interface import RegressionInterface
from rumpy.sr_tools.helper_functions import create_dir_if_empty
from rumpy.shared_framework.configuration import constants as sconst
from rumpy.regression.models.contrastive_learning import register_metadata, partition_metadata, class_retrieval
from rumpy.image_tools.blur.real_esrgan_blur import select_specific_kernel
from rumpy.shared_framework.configuration.gpu_check import device_selector

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn import cluster, metrics
import umap
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import os
from prefetch_generator import BackgroundGenerator
from collections import defaultdict
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
import textwrap


def multi_index_converter(ind, images_per_row=None):
    if images_per_row is None:
        return ind
    else:
        return int(ind / images_per_row), ind % images_per_row  # converts indices to double


# TODO: make plotting of both embedding + q optional (can choose either one)
class ContrastiveEval:
    def __init__(self, gpu='single', sp_gpu=0, q_length=256):

        # variable initialization
        self.data_handlers = []
        self.valid_metadata = []
        self.decision_mags = []
        self.total_classes = 0
        self.data_loader = None
        self.model = None
        self.metadata_len = None
        self.degradation_params = None
        self.data_encodings = None
        self.data_q = None
        self.data_dropdown_q = None
        self.embedding_length = None
        self.q_length = q_length
        self.base_folder = None
        self.metadata_keys = None
        self.metadata_mapping = None
        self.full_degradation_dict = {}
        self.plot_extension = 'pdf'
        self.img_dpi = None
        self.metadata_hyperparams = {}

        # TSNE
        self.tsne_encoding = None
        self.tsne_q = None
        self.tsne_parameters = {
            'perplexity': None
        }

        # UMAP
        self.umap_encoding = None
        self.umap_q = None
        self.umap_parameters = {
            'n_neighbors': None,
            'metric': None,
            'min_dist': None
        }

        self.device = device_selector(gpu, sp_gpu)

        # degradation dictionaries
        self.noise_dictionary = {
            'Colour/Poisson': 0,
            'Gray/Poisson': 1,
            'Colour/Gaussian': 2,
            'Gray/Gaussian': 3
        }

        # default colour coding ordering
        self.colour_codes = {0: 'r',
                             1: 'b',
                             2: 'g',
                             3: 'purple'}

        # default colour map ordering
        self.color_ranges = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'Greys', 'magma']

    def reset_metadata_len(self):
        self.metadata_len = None

    def initialize_output_folder(self, output_dir, name=None):
        """
        Prepares output folder for figures.
        :param output_dir: Output folder location.
        :param name: Override default name.
        :return: None
        """

        if name is None:
            name = '%s_%s' % (self.model.experiment, self.model.model_epoch)
        self.base_folder = os.path.join(output_dir, name)
        create_dir_if_empty(self.base_folder)

    def config_output_plots(self, file_extension='pdf', dpi=100):
        self.plot_extension = file_extension
        if file_extension == 'pdf':
            self.img_dpi = None
        else:
            self.img_dpi = dpi

    def register_metadata(self, data_handler, labelling_strategy):
        # TODO: add in labelling strategy choice
        self.metadata_len = len(data_handler.metadata_keys)

        processed_keys = register_metadata(data_handler.metadata_keys)

        self.metadata_mapping = {key: processed_keys.index(key) for key in processed_keys}
        self.metadata_keys = processed_keys
        self.valid_metadata, self.decision_mags, self.total_classes = partition_metadata(self.metadata_mapping,
                                                                                         labelling_strategy=labelling_strategy)

    def register_dataset(self, lr_data_folder, additive=False, labelling_strategy='default',
                         register_hyperparams=False):
        """
        Adds or replaces a dataset in the current list.
        :param lr_data_folder: Data folder locations (str or list)
        :param additive: Set to true to only add data, not replace.
        :param labelling_strategy: Set the precision of the labels.
        :param register_hyperparams: Set to true to also extract hyperparameters of dataset degradations.
        :return: None
        """

        if isinstance(lr_data_folder, str):
            lr_data_folder = [lr_data_folder]

        data_handlers = []
        for folder in lr_data_folder:
            rgb_handler = SuperResImages(folder,
                                         degradation_metadata_file=os.path.join(folder, 'degradation_metadata.csv'),
                                         y_only=False, split='all', input='unmodified',
                                         augmentation_normalization=False,
                                         colorspace='rgb', conv_type='jpg', scale=4)
            if self.metadata_len is None:
                self.register_metadata(rgb_handler, labelling_strategy)
            else:
                if self.metadata_len != len(rgb_handler.metadata_keys):
                    raise RuntimeError('Different types of data provided.')
            data_handlers.append(rgb_handler)

        if additive:
            self.data_handlers.extend(data_handlers)
        else:
            self.data_handlers = data_handlers

        if register_hyperparams:
            self.register_hyperparams(lr_data_folder[0])

        self.enqueue_data()

    def register_hyperparams(self, lr_data_folder):
        hyperparams_file = os.path.join(lr_data_folder, 'degradation_hyperparameters.csv')
        hyper_params = pd.read_csv(hyperparams_file, index_col=0)

        hyper_params = hyper_params.to_dict('list')  # TODO: expand this to multiple degradations of the same type...

        for index in range(len(hyper_params['hyperparam'])):
            if 'sigma_x_range' in hyper_params['hyperparam'][index] or 'sigma_y_range' in hyper_params['hyperparam'][
                index]:
                v1, v2 = hyper_params['value'][index][1:-1].split(',')
                val = (float(v1), float(v2))
            else:
                val = hyper_params['value'][index]

            self.metadata_hyperparams[hyper_params['hyperparam'][index]] = val

    def enqueue_data(self, num_threads=0):
        """
        Transfers datasets to data loaders.
        :param num_threads: Number of parallel threads to setup for dataloader.  Default none.
        :return: None
        """
        if len(self.data_handlers) == 0:
            self.data_loader = DataLoader(self.data_handlers[0], batch_size=1, num_workers=num_threads)
        else:
            self.data_loader = DataLoader(ConcatDataset(self.data_handlers), batch_size=1, num_workers=num_threads)

    def define_embedding_length(self, embedding_length):
        self.embedding_length = embedding_length

    def define_model(self, base_folder, name, epoch, gpu='single', sp_gpu=0):
        """
        Defines contrastive model.
        :param base_folder: Model location.
        :param name: Model name.
        :param epoch: Epoch to evaluate.
        :param gpu: Set to 'single' to use one gpu, and 'off' to use CPU.
        :param sp_gpu: Specific GPU to use.
        :return: None
        """

        if isinstance(epoch, str):
            load_epoch = int(epoch) if epoch.isnumeric() else epoch
        else:
            load_epoch = epoch

        self.model = RegressionInterface(base_folder, name,
                                         load_epoch=load_epoch, gpu=gpu, sp_gpu=sp_gpu)

        if self.model.model.regressor_type != 'contrastive':
            raise RuntimeError('Only contrastive models can be used with this eval system.')

        self.define_embedding_length(self.model.model.get_embedding_len())

    def generate_data_encoding(self, run_tsne=True, run_umap=False, data_loader=None, model=None,
                               max_images=None, has_dropdown=False, dropdown_size=1, **kwargs):
        """
        Run model through entire dataset, generating encodings for each image.
        :param run_tsne: Set to true to run TSNE on results immediately.
        :param run_umap: Set to true to run UMAP on results immediately.
        :param data_loader: (optional) Dataloader with data to encode.  If not provided, will use internally
        defined dataloader.
        :param model: (optional) Model to run on data.  If not provided, will use internally defined model.
        :param truncate: Set the maximum number of images to process.
        :param max_images: Set maximum number of images to load and process.
        :return: None
        """

        if model is None:
            if self.model is None:
                raise RuntimeError('Model must be provided or pre-initialized to generate encodings.')
            selected_model = self.model
        else:
            selected_model = model

        if data_loader is None:
            selected_dataloader = self.data_loader
            if self.data_loader is None:
                if len(self.data_handlers) > 0:
                    raise RuntimeError('Data must be provided or pre-initialized to generate encodings.')
                self.enqueue_data()
        else:
            selected_dataloader = data_loader

        if max_images is not None:
            array_length = max_images
        else:
            array_length = len(selected_dataloader)

        self.degradation_params = np.zeros((array_length, self.metadata_len))
        self.data_encodings = np.zeros((array_length, self.embedding_length))
        self.data_q = np.zeros((array_length, self.q_length))
        self.data_dropdown_q = np.zeros((array_length, dropdown_size))
        self.image_names = []

        image_names = []

        # for index, batch in tqdm(enumerate(selected_dataloader), total=len(selected_dataloader)):
        with tqdm(total=len(selected_dataloader)) as pbar_val:
            prefetcher = BackgroundGenerator(selected_dataloader)
            for index, batch in enumerate(prefetcher):
                image = batch['lr'].to(device=self.device)
                self.image_names.append(batch['tag'][0])
                if 'metadata' in batch:
                    self.degradation_params[index, :] = batch['metadata'].numpy()
                else:
                    self.degradation_params[index, :] = batch['target_metadata'].numpy()

                encoded_rep, _, _ = selected_model.net_run_and_process(image)

                self.data_encodings[index, :] = encoded_rep[0].cpu().numpy()

                if has_dropdown:
                    self.data_q[index, :] = encoded_rep[1]['q'].cpu().numpy()
                    self.data_dropdown_q[index, :] = encoded_rep[1]['dropdown_q'].cpu().numpy()
                else:
                    self.data_q[index, :] = encoded_rep[1].cpu().numpy()

                image_names.append(batch['tag'])  # collects image names if required
                if max_images is not None and index == max_images - 1:
                    print('Images capped at %s' % max_images)
                    break
                pbar_val.update(1)

        self.interpret_metadata()

        if run_tsne:
            self.fit_tsne(**kwargs)

        if run_umap:
            self.fit_umap(**kwargs)

        return image_names

    def interpret_metadata(self):
        """
        Function that interprets dataset metadata into labels, degradation magnitudes etc.
        TODO: other degradations need to fit here
        """

        noise_labels = []
        combined_noise_compression_labels = []
        noise_magnitudes = []
        compression_magnitudes = []
        compression_classes = []
        blur_params = defaultdict(list)

        m_map = self.metadata_mapping

        # TODO: remove all the extra classes and based them all on the training class, need to adjust plots to suits

        for i, param_vector in enumerate(self.degradation_params):
            training_class = class_retrieval(param_vector, self.valid_metadata, m_map, self.decision_mags,
                                             self.total_classes)

            if 'noise' in self.valid_metadata:  # manual noise class labelling
                if param_vector[m_map['gaussian_noise_scale']] > 0 and param_vector[m_map['gray_noise_boolean']] == 0:
                    noise_labels.append('Colour/Gaussian')
                elif param_vector[m_map['gray_noise_boolean']] == 0 and param_vector[m_map['poisson_noise_scale']] > 0:
                    noise_labels.append('Colour/Poisson')
                elif param_vector[m_map['gaussian_noise_scale']] > 0 and param_vector[m_map['gray_noise_boolean']] == 1:
                    noise_labels.append('Gray/Gaussian')
                elif param_vector[m_map['gray_noise_boolean']] == 1 and param_vector[m_map['poisson_noise_scale']] > 0:
                    noise_labels.append('Gray/Poisson')

                if param_vector[m_map['gaussian_noise_scale']] > 0:  # magnitude extraction
                    noise_magnitudes.append(param_vector[m_map['gaussian_noise_scale']])
                else:
                    noise_magnitudes.append(param_vector[m_map['poisson_noise_scale']])

            if 'compression' in self.valid_metadata:
                if ('jpeg_quality_factor' in m_map and param_vector[
                    m_map['jpeg_quality_factor']] > 0) or 'jm_qpi' not in m_map:
                    c_class = 'jpeg'
                    c_mag = param_vector[m_map['jpeg_quality_factor']]
                else:
                    c_class = 'jm'
                    c_mag = param_vector[m_map['jm_qpi']]

                compression_magnitudes.append(c_mag)
                compression_classes.append(c_class)

            if 'blur' in self.valid_metadata:
                blur_params['sigma_x'].append(param_vector[m_map['sigma_x']])
                blur_params['sigma_y'].append(param_vector[m_map['sigma_y']])
                blur_params['rotation'].append(param_vector[m_map['rotation']])
                blur_params['kernel_type'].append(param_vector[m_map['kernel_type']])
                blur_params['beta_p'].append(param_vector[m_map['beta_p']])
                blur_params['beta_g'].append(param_vector[m_map['beta_g']])
                blur_params['omega_c'].append(param_vector[m_map['omega_c']])

            if 'noise' in self.valid_metadata and 'compression' in self.valid_metadata and 'jpeg_quality_factor' in m_map:
                class_label = ''
                if param_vector[m_map['gray_noise_boolean']] > 0:
                    class_label += 'gray_'
                else:
                    class_label += 'colour_'

                if param_vector[m_map['gaussian_noise_scale']] > 0:
                    if param_vector[m_map['gaussian_noise_scale']] > 0.5:
                        class_label += 'high_gauss_'
                    else:
                        class_label += 'low_gauss_'
                else:
                    if param_vector[m_map['poisson_noise_scale']] > 0.5:
                        class_label += 'high_poisson_'
                    else:
                        class_label += 'low_poisson_'

                if param_vector[m_map['jpeg_quality_factor']] > 0.5:
                    class_label += 'low_compression'
                else:
                    class_label += 'high_compression'

                combined_noise_compression_labels.append(class_label)

        noise_magnitudes = np.array(noise_magnitudes)
        compression_magnitudes = np.array(compression_magnitudes)

        self.full_degradation_dict = {  # final dictionary with all metadata
            'noise': [noise_labels, noise_magnitudes],
            'compression': [compression_magnitudes, compression_classes],
            'combined_noise_compression': combined_noise_compression_labels,
            'blur_params': blur_params
        }

    def fit_tsne(self, perplexity=40.0, normalize_fit=False, **kwargs):
        """
        Run TSNE on both the model embedding and Q projection.
        """
        self.tsne_parameters['perplexity'] = perplexity

        self.tsne_encoding = TSNE(n_components=2, init='pca', perplexity=perplexity).fit_transform(self.data_encodings)
        self.tsne_q = TSNE(n_components=2, init='pca', perplexity=perplexity).fit_transform(self.data_q)
        if normalize_fit:
            self.tsne_encoding = (self.tsne_encoding - np.min(self.tsne_encoding)) / np.ptp(self.tsne_encoding)
            self.tsne_q = (self.tsne_q - np.min(self.tsne_q)) / np.ptp(self.tsne_q)

    def fit_umap(self, n_neighbors=25, metric='cosine', min_dist=0.5, normalize_fit=False, **kwargs):
        """
        Run UMAP on both the model embedding and Q projection.
        """
        self.umap_parameters['n_neighbors'] = n_neighbors
        self.umap_parameters['metric'] = metric
        self.umap_parameters['min_dist'] = min_dist

        self.umap_encoding = umap.UMAP(n_neighbors=n_neighbors, metric=metric, min_dist=min_dist).fit_transform(
            self.data_encodings)
        self.umap_q = umap.UMAP(n_neighbors=n_neighbors, metric=metric, min_dist=min_dist).fit_transform(self.data_q)
        if normalize_fit:
            self.umap_encoding = (self.umap_encoding - np.min(self.umap_encoding)) / np.ptp(self.umap_encoding)
            self.umap_q = (self.umap_q - np.min(self.umap_q)) / np.ptp(self.umap_q)

    def export_encoding(self):
        # TODO: complete
        pass

    def multiplot_finishing(self, fig, ax, titles, filename, images_per_row=None, suptitle=None, legend=None,
                            legend_handles=None):
        """
        Applies final finishing touches to subplots and saves to file.
        :param fig: Matplotlib figure.
        :param ax: Subplot axes.
        :param titles: Individual title for each axis.
        :param filename: Output filename (also include extension).
        :param images_per_row: Images per row (required for multi-row/column subplots).
        :param suptitle: Any super-title to add.
        :param legend: Set to true to include legend.
        :param legend_handles: Any custom handles to use for legend.  Set individual entries to None to skip for specific axes.
        :return: None
        """

        # TODO: add optional direct plot viewing (plt.show())
        for i, title in enumerate(titles):
            ax[multi_index_converter(i, images_per_row)].set_title(title, fontsize=20)
            if legend:
                if legend_handles is None or legend_handles[i] is None:
                    ax[multi_index_converter(i, images_per_row)].legend()
                else:
                    ax[multi_index_converter(i, images_per_row)].legend(handles=legend_handles[i])

        if suptitle:
            plt.suptitle(suptitle, fontsize=24)

        fig.tight_layout()
        fig.savefig(os.path.join(self.base_folder, filename + '.%s' % self.plot_extension), dpi=self.img_dpi)
        plt.close(fig)

    def representation_selector(self, rep_type='tsne'):
        """
        Selects between TSNE or UMAP for plotting methods.
        """
        if rep_type.lower() == 'tsne':
            rep_encoding = self.tsne_encoding
            rep_q = self.tsne_q
            plot_title = 'TSNE | ' + '  '.join(
                ['%s: %s' % (k.capitalize(), str(v)) for k, v in self.tsne_parameters.items()])
            plot_title = textwrap.fill(plot_title, 30)
            save_file_label = 'TSNE'

        elif rep_type.lower() == 'umap':
            rep_encoding = self.umap_encoding
            rep_q = self.umap_q
            plot_title = 'UMAP | ' + ' | '.join(
                ['%s: %s' % (k.capitalize(), str(v)) for k, v in self.umap_parameters.items()])
            plot_title = textwrap.fill(plot_title, 30)
            save_file_label = 'UMAP'
        else:
            raise RuntimeError('Requested representation not available.')

        return rep_encoding, rep_q, plot_title, save_file_label

    def clustering_score_titles(self, rep_encoding, rep_q, titles, cluster_classes, db_score, ch_score, sil_score):
        num_unique_classes = len(set(cluster_classes))

        if db_score:
            db_score_encoding = metrics.davies_bouldin_score(rep_encoding, cluster_classes)
            db_score_q = metrics.davies_bouldin_score(rep_q, cluster_classes)

            titles[0] = titles[0] + '\nDavies-Bouldin Score (%d classes) \u2193: %.4f' % (
            num_unique_classes, db_score_encoding)
            titles[1] = titles[1] + '\nDavies-Bouldin Score (%d classes) \u2193: %.4f' % (
            num_unique_classes, db_score_q)

        if ch_score:
            ch_score_encoding = metrics.calinski_harabasz_score(rep_encoding, cluster_classes)
            ch_score_q = metrics.calinski_harabasz_score(rep_q, cluster_classes)

            titles[0] = titles[0] + '\nCalinski-Harabasz Score (%d classes) \u2191: %.4f' % (
            num_unique_classes, ch_score_encoding)
            titles[1] = titles[1] + '\nCalinski-Harabasz Score (%d classes) \u2191: %.4f' % (
            num_unique_classes, ch_score_q)

        if sil_score:
            sil_score_encoding = metrics.silhouette_score(rep_encoding, cluster_classes, metric='euclidean')
            sil_score_q = metrics.silhouette_score(rep_q, cluster_classes, metric='euclidean')

            titles[0] = titles[0] + '\nSilhouette Coefficient (%d classes) \u2191: %.4f' % (
            num_unique_classes, sil_score_encoding)
            titles[1] = titles[1] + '\nSilhouette Coefficient (%d classes) \u2191: %.4f' % (
            num_unique_classes, sil_score_q)

        return titles

    def plot_no_labels(self, rep_type='tsne'):
        """
        Plots TSNE/UMAP results directly, adding no annotations.
        """
        rep_encoding, rep_q, plot_title, save_file_label = self.representation_selector(rep_type=rep_type)

        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        ax[0].scatter(rep_encoding[:, 0], rep_encoding[:, 1])
        ax[1].scatter(rep_q[:, 0], rep_q[:, 1])

        self.multiplot_finishing(fig, ax, ['Encoding ' + plot_title, 'Q ' + plot_title],
                                 'no_labels_' + save_file_label)

    def plot_compression(self, rep_type='tsne', db_score=False, ch_score=False, sil_score=False):
        """
        Plots the TSNE/UMAP results with colour-coding for JPEG quality factor size.
        """
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))

        rep_encoding, rep_q, plot_title, save_file_label = self.representation_selector(rep_type=rep_type)

        compression_magnitudes = self.full_degradation_dict['compression'][0]
        compression_classes = self.full_degradation_dict['compression'][1]
        colour_tracker = set()
        compression_class_dict = {'jpeg': 0,
                                  'jm': 1}

        for i in range(rep_encoding.shape[0]):
            compression_class = compression_classes[i]
            colour_type = self.color_ranges[compression_class_dict[compression_class]]
            colour_tracker.add(compression_class)

            scat_1 = ax[0].scatter(rep_encoding[i, 0], rep_encoding[i, 1],
                                   vmin=0, vmax=1,
                                   c=compression_magnitudes[i],
                                   cmap=colour_type)

            scat_2 = ax[1].scatter(rep_q[i, 0], rep_q[i, 1],
                                   vmin=0, vmax=1,
                                   c=compression_magnitudes[i],
                                   cmap=colour_type)

        fig.colorbar(scat_1, ax=ax[0])  # TODO: need to translate these into actual compression values
        fig.colorbar(scat_2, ax=ax[1])

        handles = []
        for label in colour_tracker:  # manually establishing legend entries
            cmap = self.color_ranges[compression_class_dict[label]]
            colour = cm.get_cmap(cmap)(0.6)
            handles.append(mlines.Line2D([], [],
                                         color=colour,
                                         marker='o',
                                         linestyle='None',
                                         label=label))
        handles = [handles] * 2

        titles = [plot_title, plot_title]

        if db_score or ch_score or sil_score:
            cluster_classes = [compression_class_dict[compression_classes[i]] for i, _ in
                               enumerate(compression_classes)]
            if len(set(cluster_classes)) > 1:
                titles = self.clustering_score_titles(rep_encoding, rep_q, titles, cluster_classes, db_score, ch_score,
                                                      sil_score)

        self.multiplot_finishing(fig, ax, ['Encoding ' + titles[0], 'Q ' + titles[1]],
                                 'compression_' + save_file_label, legend=True, legend_handles=handles)

    def plot_blur(self, rep_type='tsne', forced_axis_lim=None, filename_tag=None):
        """
        Plots the TSNE/UMAP results with colour-coding for blur kernel features.
        """
        fig, ax = plt.subplots(2, 2, figsize=(18, 14))

        rep_encoding, rep_q, plot_title, save_file_label = self.representation_selector(rep_type=rep_type)

        blur_x = self.full_degradation_dict['blur_params']['sigma_x']
        blur_y = self.full_degradation_dict['blur_params']['sigma_y']
        blur_type = self.full_degradation_dict['blur_params']['kernel_type']
        blur_rot = self.full_degradation_dict['blur_params']['rotation']
        blur_betap = self.full_degradation_dict['blur_params']['beta_p']
        blur_betag = self.full_degradation_dict['blur_params']['beta_g']
        blur_omegac = self.full_degradation_dict['blur_params']['omega_c']

        colour_tracker = set()

        for i in range(rep_encoding.shape[0]):
            colour_type = self.color_ranges[int(blur_type[i])]
            colour_tracker.add(sconst.blur_kernel_code_conversion[blur_type[i]])

            colour_mag = max(blur_x[i], blur_y[i])

            scat_1 = ax[0, 0].scatter(rep_encoding[i, 0],
                                      rep_encoding[i, 1],
                                      vmin=0, vmax=1,
                                      c=colour_mag,
                                      cmap=colour_type)

            scat_2 = ax[0, 1].scatter(rep_q[i, 0],
                                      rep_q[i, 1],
                                      vmin=0, vmax=1,
                                      c=colour_mag,
                                      cmap=colour_type)

            sig_x_real = (blur_x[i] * (
                        self.metadata_hyperparams['sigma_x_range'][1] - self.metadata_hyperparams['sigma_x_range'][0]) +
                          self.metadata_hyperparams['sigma_x_range'][0])
            sig_y_real = (blur_y[i] * (
                        self.metadata_hyperparams['sigma_y_range'][1] - self.metadata_hyperparams['sigma_y_range'][0]) +
                          self.metadata_hyperparams['sigma_y_range'][0])

            kernel = select_specific_kernel(sconst.blur_kernel_code_conversion[blur_type[i]], 21,
                                            sig_x_real, sig_y_real, blur_rot[i], blur_betag[i], blur_betap[i],
                                            blur_omegac[i])

            ab_enc = AnnotationBbox(OffsetImage(kernel, zoom=1.0), (rep_encoding[i, 0], rep_encoding[i, 1]),
                                    frameon=False)
            ax[1, 0].add_artist(ab_enc)

            ab_q = AnnotationBbox(OffsetImage(kernel, zoom=1.0), (rep_q[i, 0], rep_q[i, 1]),
                                  frameon=False)
            ax[1, 1].add_artist(ab_q)

            if forced_axis_lim is not None:
                ax[0, 0].set_xlim(forced_axis_lim[0])
                ax[0, 0].set_ylim(forced_axis_lim[1])

                ax[0, 1].set_xlim(forced_axis_lim[0])
                ax[0, 1].set_ylim(forced_axis_lim[1])

            ax[1, 0].set_ylim(ax[0, 0].get_ylim())
            ax[1, 1].set_ylim(ax[0, 1].get_ylim())

            ax[1, 0].set_xlim(ax[0, 0].get_xlim())
            ax[1, 1].set_xlim(ax[0, 1].get_xlim())

        handles = []
        for label in colour_tracker:  # manually establishing legend entries
            cmap = self.color_ranges[sconst.blur_kernel_code_conversion[label]]
            colour = cm.get_cmap(cmap)(0.6)
            handles.append(mlines.Line2D([], [],
                                         color=colour,
                                         marker='o',
                                         linestyle='None',
                                         label=label))
        handles = [handles] * 2 + [None, None]

        # fig.colorbar(scat_1, ax=ax[0, 0])  # TODO: indicate that these are actual blur kernel sigmas
        # fig.colorbar(scat_2, ax=ax[0, 1])

        if filename_tag is None:
            file_name = 'blur_kernel_' + save_file_label
        else:
            file_name = 'blur_kernel_' + save_file_label + '_' + filename_tag

        self.multiplot_finishing(fig, ax,
                                 ['Encoding ' + plot_title, 'Q ' + plot_title, 'Encoding Kernel View', 'Q Kernel View'],
                                 file_name, legend_handles=handles, images_per_row=2, legend=True)

    def plot_noise(self, plot_magnitudes=False, rep_type='tsne', db_score=False, ch_score=False, sil_score=False):
        """
        Plots the TSNE/UMAP results with colour-coding for the noise type.
        :param plot_magnitudes: Set to true to print additional plots with the noise magnitude indicated by colour.
        """

        rep_encoding, rep_q, plot_title, save_file_label = self.representation_selector(rep_type=rep_type)

        if plot_magnitudes:
            fig, ax = plt.subplots(2, 2, figsize=(18, 12))
            images_per_row = 2
            titles = ['Encoding ' + plot_title, 'Q ' + plot_title, 'Noise Magnitudes', 'Noise Magnitudes']
            filename = 'noise_' + save_file_label + '_with_magnitudes'
            handles = [None, None]
        else:
            fig, ax = plt.subplots(1, 2, figsize=(18, 6))
            images_per_row = None
            titles = ['Encoding ' + plot_title, 'Q ' + plot_title]
            filename = 'noise_' + save_file_label
            handles = None

        noise_labels = self.full_degradation_dict['noise'][0]

        for _, label in enumerate(np.unique(noise_labels)):
            sel_indices = [l == label for l in noise_labels]
            n_code = self.noise_dictionary[label]
            ax[multi_index_converter(0, images_per_row)].scatter(rep_encoding[sel_indices, 0],
                                                                 rep_encoding[sel_indices, 1],
                                                                 c=[self.colour_codes[n_code]] * sum(sel_indices),
                                                                 label=label)
            ax[multi_index_converter(1, images_per_row)].scatter(rep_q[sel_indices, 0],
                                                                 rep_q[sel_indices, 1],
                                                                 c=[self.colour_codes[n_code]] * sum(sel_indices),
                                                                 label=label)

        if db_score or ch_score or sil_score:
            cluster_classes = [self.noise_dictionary[label] for label in noise_labels]
            if len(set(cluster_classes)) > 1:
                titles = self.clustering_score_titles(rep_encoding, rep_q, titles, cluster_classes, db_score, ch_score,
                                                      sil_score)

        if plot_magnitudes:
            noise_magnitudes = self.full_degradation_dict['noise'][1]

            colour_tracker = set()

            for i in range(rep_encoding.shape[0]):
                colour_type = self.color_ranges[self.noise_dictionary[noise_labels[i]]]
                colour_tracker.add(noise_labels[i])
                scat_1 = ax[multi_index_converter(2, images_per_row)].scatter(rep_encoding[i, 0],
                                                                              rep_encoding[i, 1],
                                                                              vmin=0, vmax=1,
                                                                              c=noise_magnitudes[i],
                                                                              cmap=colour_type)

                scat_2 = ax[multi_index_converter(3, images_per_row)].scatter(rep_q[i, 0],
                                                                              rep_q[i, 1],
                                                                              vmin=0, vmax=1,
                                                                              c=noise_magnitudes[i],
                                                                              cmap=colour_type)

            mag_handles = []
            for label in colour_tracker:  # manually establishing legend entries
                cmap = self.color_ranges[self.noise_dictionary[label]]
                colour = cm.get_cmap(cmap)(0.6)
                mag_handles.append(mlines.Line2D([], [],
                                                 color=colour,
                                                 marker='o',
                                                 linestyle='None',
                                                 label=label))
            handles.extend([mag_handles] * 2)

            # TODO: directly placing colorbars makes plots look ugly - need to fix,
            #  also need to translate 1-0 into actual noise values
            # fig.colorbar(scat_1, ax=ax[multi_index_converter(2, images_per_row)])
            # fig.colorbar(scat_2, ax=ax[multi_index_converter(3, images_per_row)])

        self.multiplot_finishing(fig, ax, titles, filename, legend=True, images_per_row=images_per_row,
                                 legend_handles=handles)

    def plot_combined_noise_compression(self, rep_type='tsne'):
        """
        Plots the TSNE/UMAP results with colour-coding for the noise type and marker size linked to JPEG quality factor.
        """

        rep_encoding, rep_q, _, save_file_label = self.representation_selector(rep_type=rep_type)

        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        noise_labels = self.full_degradation_dict['noise'][0]
        compression_magnitudes = self.full_degradation_dict['compression'][0]

        colour_tracker = set()
        for i in range(rep_encoding.shape[0]):
            n_code = self.noise_dictionary[noise_labels[i]]
            colour_tracker.add(noise_labels[i])
            scat_size = compression_magnitudes[i] * 30

            ax[0].scatter(rep_encoding[i, 0],
                          rep_encoding[i, 1],
                          c=self.colour_codes[n_code],
                          s=scat_size)
            ax[1].scatter(rep_q[i, 0],
                          rep_q[i, 1],
                          c=self.colour_codes[n_code],
                          s=scat_size)

        mag_handles = []
        handles = []
        for label in colour_tracker:  # manually establishing legend entries
            colour = self.colour_codes[self.noise_dictionary[label]]
            mag_handles.append(mlines.Line2D([], [],
                                             color=colour,
                                             marker='o',
                                             linestyle='None',
                                             label=label))
        handles.extend([mag_handles] * 2)

        self.multiplot_finishing(fig, ax, ['Embedding Multi-Plot', 'Q Multi-Plot'],
                                 'combined_noise_compression_' + save_file_label,
                                 legend=True, legend_handles=handles)

    def plot_16_class_noise_compression(self, rep_type='tsne'):
        """
        Plots the TSNE/UMAP results with colour-coding for the standard 16 classes used for noise/compression.
        """
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))

        rep_encoding, rep_q, _, save_file_label = self.representation_selector(rep_type=rep_type)

        combined_labels = self.full_degradation_dict['combined_noise_compression']
        images_per_row = None

        cm = plt.get_cmap('tab20')

        num_colors = len(np.unique(combined_labels))

        for posn, label in enumerate(np.unique(combined_labels)):
            sel_indices = [l == label for l in combined_labels]
            ax[multi_index_converter(0, images_per_row)].scatter(rep_encoding[sel_indices, 0],
                                                                 rep_encoding[sel_indices, 1],
                                                                 c=[cm(1. * posn / num_colors)] * sum(sel_indices),
                                                                 label=label)
            ax[multi_index_converter(1, images_per_row)].scatter(rep_q[sel_indices, 0],
                                                                 rep_q[sel_indices, 1],
                                                                 c=[cm(1. * posn / num_colors)] * sum(sel_indices),
                                                                 label=label)

        self.multiplot_finishing(fig, ax, ['Encoding %s' % rep_type.upper(), 'Q %s' % rep_type.upper()],
                                 'combined_16_classes_%s' % save_file_label, legend=True)

    def plot_blur_noise_compression_two_column(self, rep_type='tsne', forced_axis_lim=None,
                                               db_score=False, ch_score=False, sil_score=False,
                                               include_compression=True):
        """
        Plots the TSNE/UMAP results with blur, noise and compression.
        """
        if include_compression:
            fig, ax = plt.subplots(4, 2, figsize=(27, 30))
        else:
            fig, ax = plt.subplots(3, 2, figsize=(27, 30))

        rep_encoding, rep_q, plot_title, save_file_label = self.representation_selector(rep_type=rep_type)

        all_handles = []
        all_titles = []
        final_file_name = ''

        # BLUR
        blur_x = self.full_degradation_dict['blur_params']['sigma_x']
        blur_y = self.full_degradation_dict['blur_params']['sigma_y']
        blur_type = self.full_degradation_dict['blur_params']['kernel_type']
        blur_rot = self.full_degradation_dict['blur_params']['rotation']
        blur_betap = self.full_degradation_dict['blur_params']['beta_p']
        blur_betag = self.full_degradation_dict['blur_params']['beta_g']
        blur_omegac = self.full_degradation_dict['blur_params']['omega_c']

        colour_tracker = set()

        for i in range(rep_encoding.shape[0]):
            colour_type = self.color_ranges[int(blur_type[i])]
            colour_tracker.add(sconst.blur_kernel_code_conversion[blur_type[i]])

            colour_mag = max(blur_x[i], blur_y[i])

            scat_1 = ax[0, 0].scatter(rep_encoding[i, 0],
                                      rep_encoding[i, 1],
                                      vmin=0, vmax=1,
                                      c=colour_mag,
                                      cmap=colour_type)

            scat_2 = ax[0, 1].scatter(rep_q[i, 0],
                                      rep_q[i, 1],
                                      vmin=0, vmax=1,
                                      c=colour_mag,
                                      cmap=colour_type)

            sig_x_real = (blur_x[i] * (
                        self.metadata_hyperparams['sigma_x_range'][1] - self.metadata_hyperparams['sigma_x_range'][0]) +
                          self.metadata_hyperparams['sigma_x_range'][0])
            sig_y_real = (blur_y[i] * (
                        self.metadata_hyperparams['sigma_y_range'][1] - self.metadata_hyperparams['sigma_y_range'][0]) +
                          self.metadata_hyperparams['sigma_y_range'][0])

            kernel = select_specific_kernel(sconst.blur_kernel_code_conversion[blur_type[i]], 21,
                                            sig_x_real, sig_y_real, blur_rot[i], blur_betag[i], blur_betap[i],
                                            blur_omegac[i])

            ab_enc = AnnotationBbox(OffsetImage(kernel, zoom=1.0), (rep_encoding[i, 0], rep_encoding[i, 1]),
                                    frameon=False)
            ax[1, 0].add_artist(ab_enc)

            ab_q = AnnotationBbox(OffsetImage(kernel, zoom=1.0), (rep_q[i, 0], rep_q[i, 1]),
                                  frameon=False)
            ax[1, 1].add_artist(ab_q)

            if forced_axis_lim is not None:
                ax[0, 0].set_xlim(forced_axis_lim[0])
                ax[0, 0].set_ylim(forced_axis_lim[1])

                ax[0, 1].set_xlim(forced_axis_lim[0])
                ax[0, 1].set_ylim(forced_axis_lim[1])

            ax[1, 0].set_ylim(ax[0, 0].get_ylim())
            ax[1, 1].set_ylim(ax[0, 1].get_ylim())

            ax[1, 0].set_xlim(ax[0, 0].get_xlim())
            ax[1, 1].set_xlim(ax[0, 1].get_xlim())

        handles = []
        for label in colour_tracker:  # manually establishing legend entries
            cmap = self.color_ranges[sconst.blur_kernel_code_conversion[label]]
            colour = cm.get_cmap(cmap)(0.6)
            handles.append(mlines.Line2D([], [],
                                         color=colour,
                                         marker='o',
                                         linestyle='None',
                                         label=label))

        # Update handles, titles and file name
        all_handles.extend([handles, handles, None, None])
        all_titles.extend(
            ['Blur Encoding ' + plot_title, 'Blur Q ' + plot_title, 'Encoding Kernel View', 'Q Kernel View'])
        final_file_name = final_file_name + 'blur_'

        # NOISE
        noise_titles = ['Noise Encoding ' + plot_title, 'Noise Q ' + plot_title]
        noise_labels = self.full_degradation_dict['noise'][0]
        noise_magnitudes = self.full_degradation_dict['noise'][1]

        colour_tracker = set()

        for i in range(rep_encoding.shape[0]):
            colour_type = self.color_ranges[self.noise_dictionary[noise_labels[i]]]
            colour_tracker.add(noise_labels[i])
            scat_1 = ax[2, 0].scatter(rep_encoding[i, 0],
                                      rep_encoding[i, 1],
                                      vmin=0, vmax=1,
                                      c=noise_magnitudes[i],
                                      cmap=colour_type)

            scat_2 = ax[2, 1].scatter(rep_q[i, 0],
                                      rep_q[i, 1],
                                      vmin=0, vmax=1,
                                      c=noise_magnitudes[i],
                                      cmap=colour_type)

        mag_handles = []
        for label in colour_tracker:  # manually establishing legend entries
            cmap = self.color_ranges[self.noise_dictionary[label]]
            colour = cm.get_cmap(cmap)(0.6)
            mag_handles.append(mlines.Line2D([], [],
                                             color=colour,
                                             marker='o',
                                             linestyle='None',
                                             label=label))

        if db_score or ch_score or sil_score:
            cluster_classes = [self.noise_dictionary[label] for label in noise_labels]
            if len(set(cluster_classes)) > 1:
                noise_titles = self.clustering_score_titles(rep_encoding, rep_q, noise_titles, cluster_classes,
                                                            db_score, ch_score, sil_score)

        # Update
        all_handles.extend([mag_handles, mag_handles])
        all_titles.extend(noise_titles)
        final_file_name = final_file_name + 'noise_mag_'

        if include_compression:
            # COMPRESSION
            compression_titles = ['Compression Encoding ' + plot_title, 'Compression Q ' + plot_title]

            compression_magnitudes = self.full_degradation_dict['compression'][0]
            compression_classes = self.full_degradation_dict['compression'][1]
            colour_tracker = set()
            compression_class_dict = {'jpeg': 0,
                                      'jm': 1}

            for i in range(rep_encoding.shape[0]):
                compression_class = compression_classes[i]
                colour_type = self.color_ranges[compression_class_dict[compression_class]]
                colour_tracker.add(compression_class)

                scat_1 = ax[3, 0].scatter(rep_encoding[i, 0], rep_encoding[i, 1],
                                          vmin=0, vmax=1,
                                          c=compression_magnitudes[i],
                                          cmap=colour_type)

                scat_2 = ax[3, 1].scatter(rep_q[i, 0], rep_q[i, 1],
                                          vmin=0, vmax=1,
                                          c=compression_magnitudes[i],
                                          cmap=colour_type)

            handles = []
            for label in colour_tracker:  # manually establishing legend entries
                cmap = self.color_ranges[compression_class_dict[label]]
                colour = cm.get_cmap(cmap)(0.6)
                handles.append(mlines.Line2D([], [],
                                             color=colour,
                                             marker='o',
                                             linestyle='None',
                                             label=label))

            if db_score or ch_score or sil_score:
                cluster_classes = [compression_class_dict[compression_classes[i]] for i, _ in
                                   enumerate(compression_classes)]
                if len(set(cluster_classes)) > 1:
                    compression_titles = self.clustering_score_titles(rep_encoding, rep_q, compression_titles,
                                                                      cluster_classes, db_score, ch_score, sil_score)

            # Update handles, titles and file name
            all_handles.extend([handles, handles])
            all_titles.extend(compression_titles)
            final_file_name = final_file_name + 'compression_'

        self.multiplot_finishing(fig, ax, all_titles,
                                 final_file_name + save_file_label,
                                 legend_handles=all_handles,
                                 images_per_row=2, legend=True)


if __name__ == '__main__':

    # Only used for testing, should create an individual script or jupyter notebook to run your own datasets
    data_types = ['noise_only', 'noise_compression', 'switched_noise_compression', 'noise_blur', 'blur',
                  'truncated_blur', 'full_blur', 'full_blur_minus_sinc', 'mixed_compression', 'blur_noise_compression',
                  'noise_only_correct_sequence', 'all_noise_compression', 'fixed_compression_random_noise']
    dtype = data_types[9]
    use_umap = False

    hub = ContrastiveEval()

    hub.define_model('path_to_model_folder', 'model', 'epoch')

    dtype = 'compression_custom'

    hub.register_dataset('path_to_dataset', register_hyperparams=True)

    hub.generate_data_encoding(run_umap=use_umap, normalize_fit=True, max_images=None)
    hub.initialize_output_folder('path_to_output_folder')

    # TSNE
    hub.plot_no_labels()

    if 'noise' in dtype:
        hub.plot_noise(plot_magnitudes=True, db_score=True, ch_score=True)

    if 'compression' in dtype:
        hub.plot_compression()
        # hub.plot_combined_noise_compression()
        # hub.plot_16_class_noise_compression()

    if 'blur' in dtype:
        hub.plot_blur()
        # hub.plot_blur(forced_axis_lim=[[0, 0.2], [0, 0.2]], filename_tag='focused_tsne')

    if use_umap:
        # UMAP
        hub.plot_no_labels(rep_type='umap')
        hub.plot_noise(plot_magnitudes=True, rep_type='umap')

        if 'compression' in dtype:
            hub.plot_compression(rep_type='umap')
            hub.plot_combined_noise_compression(rep_type='umap')
            hub.plot_16_class_noise_compression(rep_type='umap')
