import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import os
import numpy as np
import itertools
from collections import defaultdict, OrderedDict
import pandas as pd
from PIL import Image
from torchvision import transforms
import time
from colorama import init, Fore

from rumpy.shared_framework.models.base_interface import ImageModelInterface
from rumpy.shared_framework.models.model_helper_functions import prep_models

from rumpy.sr_tools.data_handler import SuperResImages, VideoSequenceImages
from rumpy.sr_tools.visualization import interpret_sisr_images, safe_image_save, extract_ims_from_gallery
from rumpy.sr_tools.metrics import Metrics, plot_cmc
from rumpy.sr_tools.helper_functions import create_dir_if_empty, recursive_empty_directory_check
from rumpy.image_tools.image_manipulation.image_functions import ycbcr_convert
from rumpy.shared_framework.configuration import constants as sconst

# multi_frame_methods = ['rcan3d']
init()  # colorama setup


# TODO: this is a generic evalhub system, however it contains a number of SISR specific attributes and functions
# These should be moved to the SISR/VSR folder as appropriate, and this system should be kept as clean as possible.
# otherwise, can move back to SISR eval and create new eval routines here.
class EvalHub:
    def __init__(self, hr_dir, lr_dir, model_and_epoch, results_name, gpu, metrics, id_source, data_split,
                 save_im, batch_size, face_rec_profiling, gallery_source, galleries, full_directory,
                 model_only, scale, model_loc, out_loc, no_image_comparison,
                 save_raw_features, num_image_save, use_celeba_blacklist, qpi_selection, data_attributes,
                 save_data_model_folders, gallery_ref_images, dataset_name, image_shortlist, metadata_file,
                 sp_gpu, time_models, recursive, data_type, num_frames, hr_selection, in_features,
                 run_lpips_on_gpu, lanczos_upsample, group_select, augmentation_normalization,
                 ignore_degradation_location):
        """
        Main Eval Class.  Param info available in net_eval.py.
        """

        # output folder setup
        self.out_dir = os.path.join(out_loc, results_name)
        self.eval_name = results_name
        create_dir_if_empty(self.out_dir)
        self.comparisons_dir = os.path.join(self.out_dir, 'model_comparisons')  # image comparison directory
        if not no_image_comparison:
            create_dir_if_empty(self.comparisons_dir)

        # Model prep
        experiment_names, eval_epochs = zip(*model_and_epoch)  # unpacking model info
        # TODO: scale needs to be removed from here
        self.model_bundles = prep_models(model_loc, experiment_names, eval_epochs, gpu, scale=scale, sp_gpu=sp_gpu)

        # data prep
        self.rgb_data = self.data_setup(full_directory, data_split, dataset_name, use_celeba_blacklist,
                                        metadata_file, lr_dir, data_type, hr_dir, scale, qpi_selection, data_attributes,
                                        image_shortlist, recursive, hr_selection, num_frames, batch_size, group_select,
                                        augmentation_normalization, ignore_degradation_location)

        removal_indices = []
        for m_index, model in enumerate(self.model_bundles):
            if model.model.model_name == 'qrcan':
                non_blind_degradations = model.model.metadata
                for deg in non_blind_degradations:
                    if deg not in self.rgb_data.dataset.metadata_keys:
                        removal_indices.append(m_index)
                        print(
                            '%s %s was removed, as the dataset does not have the required metadata for this model.%s' % (
                            Fore.RED, model.experiment, Fore.RESET))
                        break

        for r_index in reversed(removal_indices):
            self.model_bundles.pop(r_index)

        model_names = set()
        self.repeated_model_names = False
        for model in self.model_bundles:
            if model.experiment not in model_names:
                model_names.add(model.experiment)
            else:
                self.repeated_model_names = True
                break

        # removes all path signatures now that models have been located
        experiment_names = [os.path.basename(exp) if os.path.isdir(exp) else exp for exp in experiment_names]

        if self.repeated_model_names:
            for m_index in range(len(experiment_names)):
                experiment_names[m_index] = experiment_names[m_index] + '_' + str(
                    self.model_bundles[m_index].model_epoch)

        # preparing output folders for various models
        self.save_folders = {}
        extra_components = ['bicubic', ]  # always provide a bicubic upsample as a reference
        if lanczos_upsample:
            extra_components += ['Lanczos', ]  # additional type of image upsampling

        if save_im or model_only:  # generate individual model output folders
            for exp in experiment_names + extra_components:
                save_folder = os.path.join(self.out_dir, exp)
                self.save_folders[exp] = save_folder
                create_dir_if_empty(save_folder)

        # flag setup
        if model_only:
            self.metrics = None
        else:
            self.metrics = metrics
        self.scale = scale
        self.full_directory = full_directory  # run models through all images in given directory
        self.no_image_comparison = no_image_comparison  # set to true to prevent image collages from being generated
        self.no_metrics = model_only  # set to true to disable all metric calculations
        self.num_image_save = num_image_save  # sets max number of image comparisons to save
        self.save_data_model_folders = save_data_model_folders
        self.time_models = time_models  # Specify this to true to also time model execution
        self.images_processed = 0
        self.save_im = save_im
        self.data_type = data_type
        self.num_frames = num_frames
        self.hr_selection = hr_selection
        self.in_features = in_features
        self.lanczos_upsample = lanczos_upsample

        # face recognition setup
        self.face_rec_profiling = face_rec_profiling  # set to true include face recognition profiling
        self.gallery_source = gallery_source
        self.galleries = galleries
        self.save_raw_features = save_raw_features  # set to true to save raw face recognition features
        if gallery_ref_images is None:  # TODO: To deal with tensorflow GPU issues,  follow guide here: https://www.tensorflow.org/guide/gpu
            self.gallery_ref_image_loc = hr_dir
        else:
            self.gallery_ref_image_loc = gallery_ref_images

        # main metric calculation hub
        if self.metrics is not None:
            self.metric_hub = Metrics(metrics, id_source=id_source,
                                      vgg_gallery=os.path.join(gallery_source, galleries[0]),
                                      hr_data_loc=hr_dir, delimeter='>',
                                      lpips_device=sp_gpu if run_lpips_on_gpu else torch.device('cpu'))

    def data_setup(self, full_directory, data_split, dataset_name, use_celeba_blacklist,
                   metadata_file, lr_dir, data_type, hr_dir, scale, qpi_selection, data_attributes,
                   image_shortlist, recursive, hr_selection, num_frames, batch_size, group_select,
                   augmentation_normalization, ignore_degradation_location):

        # interpreting data type and quantity
        if full_directory:  # all images provided
            split = 'all'
            dataset = None
            custom_split = None
            blacklist = None
        else:
            if data_split is None:  # fall back to 'eval' split only, as defined in config constants
                split = 'eval'
            else:
                split = data_split  # or, use user provided split ('test' or 'eval')
            dataset = dataset_name  # identifies dataset, if provided
            custom_split = None
            blacklist = None

        # checking to see if any image metadata is available
        if metadata_file is None:
            metadata_file = os.path.join(lr_dir, 'degradation_metadata.csv')
        if not os.path.isfile(metadata_file):
            metadata_file = os.path.join(lr_dir, 'qpi_slices.csv')  # backup filename
            if not os.path.isfile(metadata_file):
                print('%sNo metadata file found.%s' % (Fore.RED, Fore.RESET))
                metadata_file = None
                requested_metadata = None
            else:
                requested_metadata = 'all'
        else:
            requested_metadata = 'all'  # read in all metadata by default (model will choose which metadata to use)

        if data_type == 'multi-frame':  # i.e. image groups representing different video frames
            data_arguments = {'lr_dir': lr_dir,  # LR images
                              'hr_dir': hr_dir,  # HR images (if available)
                              'y_only': False,  # request all channels
                              'split': split,
                              'input': 'unmodified',  # no pre-interp changes to be made
                              'dataset': dataset,
                              'colorspace': 'rgb',  # request RGB images
                              'conv_type': 'jpg',  # jpg YCBCR conversion (unused)
                              'scale': scale,
                              'custom_split': custom_split,
                              'blacklist': blacklist,
                              'qpi_selection': qpi_selection,  # request images degraded with a specific QPI
                              'degradation_metadata_file': metadata_file,
                              'metadata': requested_metadata,
                              'data_attributes': data_attributes,  # additional image info
                              'image_shortlist': image_shortlist,  # only read in specific images
                              'recursive_search': recursive,  # collect all images from base folder recursively
                              'hr_selection': hr_selection,
                              'num_frames': num_frames,
                              'group_select': group_select,
                              'augmentation_normalization': augmentation_normalization,
                              'ignore_degradation_location': ignore_degradation_location,
                              'model_type': 'multi-frame'
                              }  # TODO: this whole dict doesn't need to be redefined, some args could just pass through

            rgb_handler = VideoSequenceImages(**data_arguments)

        else:
            rgb_handler = SuperResImages(lr_dir, hr_dir, y_only=False, split=split, input='unmodified', dataset=dataset,
                                         colorspace='rgb', conv_type='jpg', scale=scale, custom_split=custom_split,
                                         blacklist=blacklist, qpi_selection=qpi_selection,
                                         degradation_metadata_file=metadata_file,
                                         group_select=group_select,
                                         augmentation_normalization=augmentation_normalization,
                                         metadata=requested_metadata, data_attributes=data_attributes,
                                         image_shortlist=image_shortlist, recursive_search=recursive,
                                         ignore_degradation_location=ignore_degradation_location)

        return DataLoader(dataset=rgb_handler, batch_size=batch_size)

    ### TODO: Generalise for arbitrary number of dimensions
    def channel_bundle_reverse(self, lr_data):
        """
        Reverse process which grouped channels for centre frame (as per hr_selection value)
        Assuming that lr_data's first dimension represents the number of batches.
        Example expected input: lr_data with dims (1, 9, 1, 33, 33) => (n_batches, n_frames, depth/extra_dim_for_3d_conv, H, W)
        :param lr_data: Tensor containing grouped images
        :return: Tensor containing only one image (the centre frame)
        """

        lr_data = torch.squeeze(lr_data,
                                2)  # Remove singleton dimension (Example image dimensions: (1, 9, 1, 33, 33) -> (batch_size, n_channels, singleton to enable 3D conv, H, W))

        # c_no = [(c*self.num_frames) + self.hr_selection for c in range(0, self.in_features)] # Channels to select (when using channel re-ordering)
        c_no = [self.hr_selection * self.in_features + c for c in range(0,
                                                                        self.in_features)]  # Channels to select (when no channel re-ordering is used); Example values: in_features=3 and hr_selection=1, such that the channels of the second frame (i.e. indices 3 to 5 inclusive, corresponding to the RGB channels) are selected and returned.

        lr_data = lr_data[:, c_no, :, :]

        return lr_data

    def _low_res_prep(self, lr_data, timing=True, upsample_function='bicubic'):
        """
        Upsamples and formats LR data for downstream metric calculations
        :param lr_data: batch of LR images (N, C, H, W)
        :param timing: Set to true to time upsampling function
        :param upsample_function: Type of upsampling function to use (either bicubic or lanczos)
        :return: interpolated images + timing info if requested
        """
        # TODO: investigate whether pytorch's upsampling is similar to that of PIL

        if upsample_function == 'bicubic':
            resample = Image.BICUBIC
        elif upsample_function == 'lanczos':
            resample = Image.LANCZOS
        else:
            raise RuntimeError('Upsampling type unknown')

        # Reverse process which grouped channels for centre frame (as per hr_selection value)
        ### TODO: This may actually be dependent upon the model being used (e.g. RCAN3D) rather than whether SISR/VSR is being done
        if self.data_type == 'multi-frame':
            lr_data = self.channel_bundle_reverse(lr_data)

        interp_data = torch.empty(*lr_data.shape[0:2], lr_data.shape[2] * self.scale, lr_data.shape[3] * self.scale)

        timings = []
        for i in range(lr_data.shape[0]):
            image = transforms.ToPILImage()(lr_data[i, ...])
            if timing:
                tic = time.perf_counter()
            resized_im = image.resize((image.width * self.scale, image.height * self.scale), resample=resample)
            if timing:
                toc = time.perf_counter()
                timings.append(toc - tic)
            interp_data[i, ...] = transforms.ToTensor()(resized_im)  # normalizes image values from 0 to 1 too
            # interp_data[i, ...] = ycbcr_convert(image, im_type='jpg', input='rgb', y_only=False)

        return interp_data, timings if timing else None

    def _high_res_prep(self, hr_data):
        """
        Converts an input image to a YCBCR image, ready for PSNR/SSIM calculations
        :param hr_data: Input RGB images (N, C, H, W)
        :return: converted YCBCR images
        """
        hr_prep = ImageModelInterface._standard_image_formatting(hr_data.numpy())
        for i in range(hr_prep.shape[0]):
            hr_prep[i, ...] = ycbcr_convert(hr_prep[i, ...], im_type='jpg', input='rgb', y_only=False)
        return hr_prep

    def register_metrics(self, query_im, ref_im, key, image_names, running_fr_package, running_metric_package):
        """
        Calculates metrics and adds them to provided metric dict.
        :param query_im: Query (LR) image
        :param ref_im: Reference (HR) image
        :param key: reference name for model used to generate LR image
        :param image_names: Image IDs
        :param running_fr_package: Running dict to which to add computed face recognition metrics
        :param running_metric_package: Running dict to which to add computed general metrics
        :return: Diagnostics data on metrics
        """
        metric_slice, mini_diag_string = self.metric_hub.run_metrics(query_im, ref_im,
                                                                     key=key,
                                                                     probe_names=image_names,
                                                                     request_raw=self.face_rec_profiling)
        for key in metric_slice.keys():
            if 'raw_FR_features' in key:
                running_fr_package[key].append(metric_slice[key])
            else:
                running_metric_package[key].append(metric_slice[key])
        return mini_diag_string

    def _generate_image_collage(self, output_package, probe_names, metrics=None, metric_slice=None, hr_rgb=None):

        metrics = metrics if metrics is not None else []
        metric_slice = metric_slice if metric_slice is not None else {}

        if not self.no_metrics and hr_rgb is not None:
            if self.metric_hub.boundary_data is not None:
                # Comparison package setup
                bounds = []
                for probe in probe_names:
                    if (probe + '.png') in self.metric_hub.boundary_data:
                        bounds.append(self.metric_hub.boundary_data[probe + '.png'])
                    else:
                        bounds.append(None)
                output_package['face_crop'] = (hr_rgb.numpy(), bounds)
            if 'VGG_FR_Rank' in self.metrics:
                output_package['Reference'] = extract_ims_from_gallery(self.metric_hub.gallery_ids,
                                                                       self.metric_hub.gallery_files,
                                                                       metric_slice['Image_ID'],
                                                                       hr_rgb.shape[2:], self.gallery_ref_image_loc)

        extra_info = defaultdict(lambda: None)
        for model in self.model_bundles:
            extra_info[model.experiment] = [['epoch', model.model_epoch]]
        # send results for saving or visualization
        interpret_sisr_images(output_package, metric_slice, metrics, self.comparisons_dir,
                              names=['image_comparison_%s.pdf' % probe_name.replace(os.path.sep, '_') for probe_name in
                                     probe_names],
                              direct_view=False, config='rgb',
                              extra_info=extra_info)

    def full_image_protocol(self):

        if self.face_rec_profiling and self.galleries is None:
            raise Exception('Gallery sources required if doing face recognition check.')

        if self.rgb_data.dataset.hr_base is None and not self.no_metrics:
            raise Exception('HR reference images need to be specified to calculate metrics.')

        metric_package = defaultdict(list)
        fr_package = defaultdict(list)

        with tqdm(total=len(self.rgb_data)) as pbar:
            for index, batch in enumerate(self.rgb_data):

                output_package = OrderedDict()
                lr_rgb, hr_rgb, im_names, hr_names = batch['lr'], batch['hr'], batch['tag'], batch['hr_tag']
                self.images_processed += len(im_names)

                # prepare metadata
                diag_string = ''
                probe_names = [im_name.split('.')[0] for im_name in list(im_names)]
                hr_names = [hr_name.split('.')[0] for hr_name in list(hr_names)]

                metric_package['Image_Name'].append(list(im_names))

                if self.metrics and hasattr(self.metric_hub, 'file_id_link'):  # face rec profiling only
                    metric_package['Image_ID'].append(
                        [self.metric_hub.file_id_link[hr_name] for hr_name in hr_names])

                interp_data, timing_info = self._low_res_prep(lr_rgb, timing=self.time_models)  # gen interp images
                if timing_info is not None and not self.no_metrics:
                    metric_package['LR%sruntime' % self.metric_hub.delimeter].append(timing_info)

                if self.save_im and self.images_processed < self.num_image_save:
                    safe_image_save(interp_data.numpy(), self.save_folders['bicubic'], im_names,
                                    config='rgb')  # saving bicubic output image

                if not self.no_metrics:
                    hr_prep = self._high_res_prep(hr_rgb)  # convert to YCBCR
                    output_package['HR'] = hr_rgb.numpy()  # save data for output collage

                lr_prep = self._high_res_prep(interp_data)  # convert to YCBCR

                output_package['LR'] = interp_data.numpy()

                # HR face recognition metrics
                if self.metrics and 'VGG_FR_Rank' in self.metrics:
                    hr_metrics, _ = self.metric_hub.run_metrics(hr_prep, key='HR', metrics=['VGG_FR_Rank'],
                                                                probe_names=hr_names,
                                                                request_raw=self.face_rec_profiling)
                    metric_package['HR%sVGG_FR_Rank' % self.metric_hub.delimeter].append(
                        hr_metrics['HR%sVGG_FR_Rank' % self.metric_hub.delimeter])
                    if self.face_rec_profiling:
                        fr_package['HR%sraw_FR_features' % self.metric_hub.delimeter].append(
                            hr_metrics['HR%sraw_FR_features' % self.metric_hub.delimeter])

                # LR metrics
                if not self.no_metrics:
                    mini_diag_string = self.register_metrics(lr_prep, hr_prep, 'LR', hr_names, fr_package,
                                                             metric_package)
                    diag_string += mini_diag_string

                # Lanczos Processing TODO: this could be combined with bicubic upsampling above
                if self.lanczos_upsample:
                    lanczos_data, lanczos_timing = self._low_res_prep(lr_rgb, timing=self.time_models,
                                                                      upsample_function='lanczos')
                    if not self.no_metrics and lanczos_timing is not None:
                        metric_package['Lanczos%sruntime' % self.metric_hub.delimeter].append(lanczos_timing)
                    if self.save_im and self.images_processed < self.num_image_save:
                        safe_image_save(lanczos_data.numpy(), self.save_folders['Lanczos'], im_names, config='rgb')

                    lanczos_prep = self._high_res_prep(lanczos_data)
                    if not self.no_metrics:
                        self.register_metrics(lanczos_prep, hr_prep, 'Lanczos', hr_names, fr_package,
                                              metric_package)
                    output_package['Lanczos'] = lanczos_data.numpy()

                # run models and gather stats
                for model in self.model_bundles:
                    if self.repeated_model_names:
                        m_id = model.experiment + '_' + str(model.model_epoch)
                    else:
                        m_id = model.experiment

                    if 'rgb' in model.configuration['colorspace']:
                        if model.configuration['input'] == 'unmodified':
                            selected_im = lr_rgb
                        else:
                            selected_im = interp_data
                    else:
                        selected_im = lr_prep  # ycbcr data

                    # For SISR methods, reverse channel bundling process to get, for example, original RGB for centre frame
                    ### TODO: Consider if just use model.model_type, which should always be initialised (so no need to check if attribute exists, making for cleaner code); drawback is that if multi-frame models have been saved without model_type  attribute, it will be considered SISR + doesn't use latest available info in model handler file (as stored in model.model) 
                    if self.data_type == 'multi-frame':
                        if (model.metadata['internal_params'][
                            'data_type'] == 'single-frame'):  # If model was not trained using multi-frame setting (i.e. SISR), reverse channel bundling process
                            selected_im = self.channel_bundle_reverse(selected_im)

                        elif (model.configuration[
                                  'model_type'] == 'single-frame'):  # If using multi-frame data (grouped channels) but model is based on an SISR model, then remove extra singleton dimension
                            selected_im = torch.squeeze(selected_im,
                                                        2)  # Remove singleton dimension (Example image dimensions: (1, 9, 1, 33, 33) -> (batch_size, n_channels, singleton to enable 3D conv, H, W))

                    rgb_im, ycbcr_im, _, timing = model.net_run_and_process(**{**batch, **{'lr': selected_im}},
                                                                            timing=self.time_models)
                    if not self.no_metrics and timing is not None:
                        if lr_rgb.shape[0] > 1:
                            timing = [timing / lr_rgb.shape[0]] * lr_rgb.shape[
                                0]  # assigns average time to each image in batch TODO: is this correct?
                        else:
                            timing = [timing]
                        metric_package['%s%sruntime' % (m_id, self.metric_hub.delimeter)].append(timing)
                    # TODO: remove double-list implementation....

                    # calculate metrics and organize diagnostics
                    if not self.no_metrics:
                        mini_diag_string = self.register_metrics(ycbcr_im, hr_prep, m_id, hr_names, fr_package,
                                                                 metric_package)

                        diag_string += mini_diag_string

                    # Save generated image
                    output_package[m_id] = rgb_im
                    if self.save_im and self.images_processed < self.num_image_save:  # TODO: very crude, must fix later
                        for im in im_names:
                            if os.sep in im:
                                create_dir_if_empty(os.path.join(self.save_folders[m_id], os.path.dirname(im)))

                        safe_image_save(rgb_im, self.save_folders[m_id],
                                        im_names, config='rgb')

                # generate image comparisons
                if not self.no_image_comparison and self.images_processed < self.num_image_save:
                    self._generate_image_collage(output_package, metrics=self.metrics,
                                                 metric_slice=None if self.no_metrics else
                                                 {key: metric_package[key][-1] for key in metric_package},
                                                 probe_names=probe_names, hr_rgb=hr_rgb)
                # update progress bar
                pbar.update(1)
                pbar.set_description(diag_string[:-2])

        self.face_recognition_calculations(fr_package, metric_package)
        if not self.no_metrics:
            self.manipulate_and_save_metrics(metric_package)

    def face_recognition_calculations(self, fr_package, metric_package):
        # FR metric compilation
        if self.face_rec_profiling:
            image_names = list(itertools.chain.from_iterable(metric_package['Image_Name']))
            for key in fr_package.keys():
                fr_package[key] = np.vstack(fr_package[key])
            if self.save_raw_features:
                raw_fr_dir = os.path.join(self.out_dir, 'raw_face_rec_features')
                create_dir_if_empty(raw_fr_dir)
                for key in fr_package.keys():
                    data = pd.DataFrame(fr_package[key])
                    data.insert(0, 'Image_Name', image_names)
                    data.set_index(['Image_Name']).to_csv(
                        os.path.join(raw_fr_dir, '%s-raw_fr_features.csv' % key.split('-')[0]))

            # calculating FR data
            cmc_data, extra_data, rank_data = self.metric_hub.multi_gallery_face_rec_check(fr_package, image_names,
                                                                                           [os.path.join(
                                                                                               self.gallery_source,
                                                                                               gallery) for gallery in
                                                                                               self.galleries])

            # managing and saving FR data
            fr_directory = os.path.join(self.out_dir, 'fr_metrics')
            create_dir_if_empty(fr_directory)
            plot_cmc(cmc_data, save_loc=fr_directory)
            av_rank_data = rank_data.copy()
            av_rank_data.columns = pd.MultiIndex.from_tuples([tuple(c.split('-')) for c in av_rank_data.columns])
            av_rank_data = self.average_multilevel_dataframe(av_rank_data)
            av_rank_data.loc['average'] = av_rank_data.mean()
            self.quick_save_csv_data([cmc_data, extra_data, rank_data, av_rank_data], directory=fr_directory,
                                     names=['cmc_fr_metrics.csv', 'extra_fr_metrics.csv', 'individual_im_ranks.csv',
                                            'average_ranks.csv'])

            # saving individual model fr results to their directory
            if self.save_data_model_folders:
                for model in self.model_bundles:
                    results_path = os.path.join(model.base_folder, 'result_outputs/fr_' + self.eval_name)
                    create_dir_if_empty(results_path)
                    model_keys = [c for c in cmc_data.columns if model.experiment in c]
                    self.quick_save_csv_data([cmc_data[model_keys], extra_data[model_keys], rank_data[model_keys]],
                                             directory=results_path,
                                             names=['cmc_fr_metrics.csv', 'extra_fr_metrics.csv',
                                                    'indiv_im_fr_metrics.csv'])

    def quick_save_csv_data(self, data, directory, names):
        for data, name in zip(data, names):
            data.to_csv(os.path.join(directory, name))

    def manipulate_and_save_metrics(self, metric_package):
        # combining all results
        for key in metric_package.keys():
            metric_package[key] = list(itertools.chain.from_iterable(metric_package[key]))

        # Pandas conversion and further calculations
        if 'Image_ID' in metric_package:
            indexes = ['Image_Name', 'Image_ID']
        else:
            indexes = ['Image_Name']
        full_results = pd.DataFrame.from_dict(metric_package).set_index(indexes)
        full_results.columns = pd.MultiIndex.from_tuples([tuple(c.split('>')) for c in full_results.columns])
        av_results = self.average_multilevel_dataframe(full_results)

        # saving to csv
        metrics_dir = os.path.join(self.out_dir, 'standard_metrics')
        create_dir_if_empty(metrics_dir)

        full_results.to_csv(os.path.join(metrics_dir, 'individual_metrics.csv'))
        av_results.to_csv(os.path.join(metrics_dir, 'average_metrics.csv'))

    def average_multilevel_dataframe(self, dataframe):
        r1 = dataframe.mean(axis=0).rename('Mean')
        r2 = dataframe.std(axis=0).rename('Std')
        results = pd.concat([r1, r2], axis=1)
        results = pd.DataFrame(results.stack()).T.stack(0).droplevel(level=0)
        return results
