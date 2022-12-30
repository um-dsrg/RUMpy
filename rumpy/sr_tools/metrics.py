import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from collections import defaultdict
import random
import pandas as pd
from torchvision import transforms
import torch
import itertools
import importlib
import lpips

keras_spec = importlib.util.find_spec("keras_vggface")
keras_vggface_available = keras_spec is not None

from sklearn import metrics
from sklearn.metrics.pairwise import distance_metrics
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import brentq
from scipy import interpolate

import rumpy.shared_framework.configuration.constants as sconst
from rumpy.image_tools.image_manipulation.image_functions import ycbcr_convert
from rumpy.sr_tools.helper_functions import get_actual_issame
from rumpy.sr_tools.stats import legacy_load_statistics
from rumpy.SISR.models.feature_extractors.VGGNets import VggFace
from rumpy.shared_framework.configuration.gpu_check import device_selector


def psnr(img1, img2, max_value=255.0):
    """
    Calculates peak signal-to-noise ratio (PSNR) between two images.
    :param img1: 2D image array (numpy or torch)
    :param img2: 2D image array (numpy or torch)
    :param max_value: maximum pixel value for arrays
    :return:
    """
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:  # TODO: probably to use something more obvious, like 'inf'
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


class Metrics:
    """
    Main metrics class that takes care of all necessary calculations for images specified.
    """
    def __init__(self, metrics, id_source=None,
                 vgg_gallery=
                 os.path.join(sconst.data_directory,
                              'celeba/png_samples/celeba_png_align/eval_vgg_features/gallery_0.npz'),
                 hr_data_loc=None, delimeter='-', lpips_device=torch.device('cpu')):
        """
        :param metrics: List of metrics to calculate.
        :param id_source: Location of image ID file (if calculating face recognition).
        :param vgg_gallery: Location of image gallery (if calculation face recognition).
        :param hr_data_loc: HR image data directory (if calculating facial PSNR).
        :param delimeter: Delimeter to use when outputting results.
        :param lpips_device: Calculate LPIPS using the specified device (default CPU).
        """

        self.metrics = metrics
        self.delimeter = delimeter

        if 'VGG_FR_Rank' in metrics:
            self.vgg_predictor = FaceRecognizer()
            self.gallery, self.gallery_ids, self.gallery_files = self.load_gallery(vgg_gallery)

        # Specific procedure for celeba
        if 'VGG_FR_Rank' in metrics and id_source is not None:
            self.file_id_link = {}
            with open(id_source, "r") as f:
                for index, line in enumerate(f):
                    self.file_id_link[line.split(' ')[0].split('.')[0]] = int(line.split(' ')[1])

        if 'face_PSNR' in metrics or 'true_face_PSNR' in metrics:
            self.boundary_data = pd.read_csv(os.path.join(hr_data_loc, 'face_boundaries_0.csv'),
                                             header=0, index_col=0, squeeze=True).dropna().astype(int).to_dict('index')
            marked = []
            for k, v in self.boundary_data.items():  # wipes off any entries with negative numbers TODO: make a function for this
                if any(v_in < 0 for v_in in v.values()):
                    marked.append(k)
            for k in marked:
                self.boundary_data.pop(k, None)
        else:
            self.boundary_data = None
        if 'LPIPS' in metrics:

            lpips_device = device_selector(True, lpips_device)

            self.lpips_loss_fn = lpips.LPIPS(net='alex').to(device=lpips_device)
            self.lpips_device = lpips_device

            if str(lpips_device) == 'cpu':
                str_device = self.lpips_device
            else:
                str_device = 'GPU ' + str(self.lpips_device)

            print('calculating LPIPS using %s' % str_device)

    @staticmethod
    def load_gallery(gallery):
        g_stack = np.load(gallery)
        return g_stack['out_stack'], g_stack['id_stack'], g_stack['file_stack']

    def run_psnr(self, im_a, im_ref, single_values=False, multichannel=False, max_value=1):
        if im_ref is None:
            raise Exception('Need a reference to calculate PSNR.')
        if single_values:
            indiv_psnr = []
            for ind in range(im_a.shape[0]):
                indiv_psnr.append(psnr(im_a[ind, 0, :, :], im_ref[ind, 0, :, :], max_value=max_value))
            return indiv_psnr
        else:
            if multichannel:
                return psnr(im_a, im_ref, max_value=max_value)
            else:
                return psnr(im_a[:, 0, :, :], im_ref[:, 0, :, :], max_value=max_value)

    def run_ssim(self, im_a, im_ref, single_values=False, multichannel=False, max_value=1):
        if im_ref is None:
            raise Exception('Need a reference to calculate SSIM.')

        if multichannel:
            im_a = im_a.transpose((0, 2, 3, 1))
            im_ref = im_ref.transpose((0, 2, 3, 1))

            ssim_vals = []
            for i in range(im_a.shape[0]):
                ssim_vals.append(ssim(im_a[i, :], im_ref[i, :], data_range=max_value, gaussian_weights=True,
                                      use_sample_covariance=False, sigma=1.5, multichannel=True))
            return sum(ssim_vals)/len(ssim_vals)

        else:
            im_a = im_a.transpose((1, 2, 3, 0))[0, :]
            im_ref = im_ref.transpose((1, 2, 3, 0))[0, :]

            if single_values:
                indiv_ssim = []
                for ind in range(im_a.shape[-1]):
                    indiv_ssim.append(ssim(im_a[..., ind], im_ref[..., ind], data_range=max_value, gaussian_weights=True,
                                           use_sample_covariance=False, sigma=1.5))
                return indiv_ssim
            else:
                return ssim(im_a, im_ref, data_range=max_value, gaussian_weights=True,
                            use_sample_covariance=False, sigma=1.5, multichannel=True)

    def run_face_PSNR(self, im_a, im_ref, probe_names, single_values=False, multichannel=False, max_value=1):
        if im_ref is None:
            raise Exception('Need a reference to calculate PSNR.')
        if probe_names is None:
            raise Exception('Need probe names to extract face boundaries')
        a_crop = np.copy(im_a)
        ref_crop = np.copy(im_ref)
        crop_area = np.zeros_like(im_a)
        for index, image in enumerate(probe_names):
            if (image + '.png') in self.boundary_data:  # TODO: synchronize the extension usage between methods...
                box = self.boundary_data[image + '.png']
                crop_area[index, :, box['top']:box['top']+box['height'], box['left']:box['left']+box['width']] = 1
            else:
                crop_area[index, ...] = 1
        a_crop = a_crop * crop_area
        ref_crop = ref_crop * crop_area

        if single_values:
            indiv_psnr = []
            for ind in range(a_crop.shape[0]):
                indiv_psnr.append(psnr(a_crop[ind, 0, :, :], ref_crop[ind, 0, :, :], max_value=max_value))
            return indiv_psnr
        else:
            if multichannel:
                return psnr(a_crop, ref_crop, max_value=max_value)
            else:
                return psnr(a_crop[:, 0, :, :], ref_crop[:, 0, :, :], max_value=max_value)

    def run_true_face_PSNR(self, im_a, im_ref, probe_names, single_values=False, multichannel=False, max_value=1):
        if im_ref is None:
            raise Exception('Need a reference to calculate PSNR.')
        if probe_names is None:
            raise Exception('Need probe names to extract face boundaries')

        indiv_psnr = []

        for index, image in enumerate(probe_names):
            if (image + '.png') in self.boundary_data:
                box = self.boundary_data[image + '.png']
                face = im_a[index, :, box['top']:box['top']+box['height'], box['left']:box['left']+box['width']]
                ref_face = im_ref[index, :, box['top']:box['top']+box['height'], box['left']:box['left']+box['width']]
            else:
                face = im_a[index, ...]
                ref_face = im_ref[index, ...]
            if multichannel:
                indiv_psnr.append(psnr(face, ref_face, max_value=max_value))
            else:
                indiv_psnr.append(psnr(face[0, ...], ref_face[0, ...], max_value=max_value))
        if single_values:
            return indiv_psnr
        else:
            return sum(indiv_psnr)/len(indiv_psnr)

    def run_VGG_fr_rank(self, im_a, probe_names, single_values=False, request_raw=False):

        if probe_names is None:
                raise Exception('Need a probe ID to evaluate face recognition performance.')

        probes = self.vgg_predictor.extract_features_from_batch(im_a)

        dist = FaceRecognizer.distance_feats(probes, self.gallery)
        probe_ids = [self.file_id_link[probe_name] for probe_name in probe_names]
        ranks = FaceRecognizer.cumulative_match(dist, mode='dist', verbose=False, probe_ids=probe_ids,
                                                gallery_ids=self.gallery_ids, quick_probe=True).tolist()
        if single_values:
            rank_data = ranks
        else:
            rank_data = sum(ranks)/len(ranks)
        if request_raw:
            return {'FR_rank': rank_data, 'raw_FR_features': probes}
        else:
            return rank_data

    def run_lpips(self, im_a, im_ref, single_values=False):

        if type(im_a) == np.ndarray:
            im_a = torch.from_numpy(im_a)
        if type(im_ref) == np.ndarray:
            im_ref = torch.from_numpy(im_ref)

        if self.lpips_device is not torch.device('cpu'):
            im_a = im_a.to(device=self.lpips_device)
            im_ref = im_ref.to(device=self.lpips_device)

        lpips_diff = self.lpips_loss_fn.forward(im_a, im_ref).detach().cpu().numpy().squeeze()

        if single_values:
            if len(lpips_diff.shape) == 0:
                return [lpips_diff]
            indiv_lpips = []
            for i in range(lpips_diff.shape[0]):
                indiv_lpips.append(lpips_diff[i])
            return indiv_lpips
        else:
            return lpips_diff

    def run_image_metric(self, metric, im_a, im_ref=None, probe_names=None,
                         single_values=False, max_value=1, request_raw=False, multichannel=False):
        """
        Main metric calculation function.
        :param metric: Metric to be calculated (string).
        :param im_a: Batch of query images.
        :param im_ref: Batch of reference images.
        :param probe_names: Query image names.
        :param single_values: Request metric result for each provided image individually.
        :param max_value: Max image pixel value.
        :param request_raw: Set to true to extract raw VGG features when calculating face recognition.
        :param multichannel: Calculate metrics using all provided channels.
        :return: Metric results.
        """
        if len(im_a.shape) == 3:  # ensure N, C, H, W format
            im_a = np.expand_dims(im_a, axis=0)

        if im_ref is not None and len(im_ref.shape) == 3:
            im_ref = np.expand_dims(im_ref, axis=0)

        if metric == 'PSNR':  # TODO: multichannel could also be relevant to single values area....
            return self.run_psnr(im_a, im_ref, single_values, multichannel, max_value)

        elif metric == 'SSIM':
            return self.run_ssim(im_a, im_ref, single_values, multichannel, max_value)

        # Calculates PSNR only on facial area (but leaves boundary area as a blank)
        elif metric == 'face_PSNR':  # TODO: of course, multiple speed-ups possible here...
            return self.run_face_PSNR(im_a, im_ref, probe_names, single_values, multichannel, max_value)

        # calculates PSNR only on facial area
        elif metric == 'true_face_PSNR':
            return self.run_true_face_PSNR(im_a, im_ref, probe_names, single_values, multichannel, max_value)

        # calculate VGG face recognition rank
        elif metric == 'VGG_FR_Rank':
            return self.run_VGG_fr_rank(im_a, probe_names, single_values, request_raw)

        # calculate LPIPS perceptual metric
        elif metric == 'LPIPS':
            return self.run_lpips(im_a, im_ref, single_values)
        else:
            raise RuntimeError('Metric not recognized')

    def run_metrics(self, images, references=None, key='',
                    metrics=None, probe_names=None, max_value=1, request_raw=False):
        """
        Function that runs multiple metrics for images specified.
        :param images: Images to evaluate.
        :param references: Reference images.
        :param key: Additional key to append to results dictionary.
        :param metrics: Metrics to calculate.
        :param probe_names: Image names.
        :param max_value: Images maximum pixel value.
        :param request_raw: Request raw face recognition vectors.
        :return: Dictionary of results, and a quick diagnostic string.
        """
        #TODO: also accept rgb images to prevent double-take for Face Rec Performance
        if metrics is None:
            metrics = self.metrics
        diag_string = ''
        output = defaultdict(list)

        for metric in metrics:  # TODO: deal with the needless ycbcr back-conversion for face FR!
            value = self.run_image_metric(metric, images, references, max_value=max_value,
                                                     probe_names=probe_names, single_values=True,
                                                     request_raw=request_raw)
            if type(value) == dict:
                for metric_key in value.keys():
                    if metric_key == 'FR_rank':
                        output['%s%s%s' % (key, self.delimeter, metric)] = value[metric_key]
                    else:
                        output['%s%s%s' % (key, self.delimeter, metric)] = value[metric_key]
            else:
                output['%s%s%s' % (key, self.delimeter, metric)] = value
            if metric.upper() == 'PSNR':
                diag_string = '{} {}: {:.4f}, '.format(key, metric, np.average(value))

        return output, diag_string

    def multi_gallery_face_rec_check(self, fr_vector_package, image_names, gallery_sources, verbose=False):
        """
        Calculates face recognition performance against multiple reference galleries.
        :param fr_vector_package: Face recognition VGG vectors for each query image.
        :param image_names: Image IDs.
        :param gallery_sources: Galleries to evaluate.
        :param verbose: Set to true to turn on extra diagnostic verbosity.
        :return: Face recognition data (cumulative match curve, AUC/EER data, rank data)
        """

        fr_metric_package = defaultdict(list)
        cmc_metric_package = {}
        rank_package = defaultdict(list)
        if type(fr_vector_package) == np.array:  # TODO: test to see if this overwrites input
            fr_vector_package = {'Query': fr_vector_package}

        gallery_specs = []
        for gallery_loc in gallery_sources:
            gallery, gallery_ids, gallery_files = self.load_gallery(gallery_loc)
            gallery_name = gallery_loc.split('/')[-1].split('.')[0]
            accepted_indices = [ind for ind, n in enumerate(image_names) if n not in gallery_files]
            if len(accepted_indices) == 0:  # TODO: perhaps a better failsafe would be more helpful
                print('%s does not have any valid images.' % gallery_name)
                continue
            probe_names = [image_names[ind].split('.')[0] for ind in accepted_indices]
            probe_ids = [self.file_id_link[probe_name] for probe_name in probe_names]

            gallery_specs.append(
                (gallery, gallery_ids, gallery_files, gallery_name, accepted_indices, probe_ids, probe_names))

        rank_package['Image_Name'] = image_names

        pbar = tqdm(fr_vector_package)
        pbar.set_description('Calculating Face Recognition Performance')
        for vector_key in pbar:
            for (gallery, gallery_ids, gallery_files, gallery_name, accepted_indices, probe_ids,
                 probe_names) in gallery_specs:
                if verbose:
                    print('%%%%%%%%%%%%%%%%%%%%')
                    print('Results for Gallery %s:' % gallery_name)
                query_name = gallery_name + '-' + vector_key.split('-')[0]
                single_cmc_package, single_fr_package, ranks = \
                    self.vgg_predictor.full_package(fr_vector_package[vector_key][accepted_indices, :],
                                                    gallery=gallery,
                                                    probes=probe_ids,
                                                    gallery_ids=gallery_ids,
                                                    query_name=query_name)
                for im in image_names:
                    im_split = im.split('.')[0]
                    if im_split in probe_names:
                        rank_package[query_name].append(ranks[probe_names.index(im_split)])
                    else:
                        rank_package[query_name].append(0)

                if len(cmc_metric_package) == 0:
                    cmc_metric_package['Rank'] = single_cmc_package[query_name+'_x']
                    fr_metric_package['Metric'] = ['AUC', 'EER']

                cmc_metric_package[query_name] = single_cmc_package[query_name+'_y']
                fr_metric_package[query_name] = [single_fr_package['AUC'], single_fr_package['EER']]

        cmc_data = pd.DataFrame.from_dict(cmc_metric_package).set_index(['Rank'])
        extra_data = pd.DataFrame.from_dict(fr_metric_package).set_index(['Metric'])
        rank_data = pd.DataFrame.from_dict(rank_package).set_index(['Image_Name'])

        return cmc_data, extra_data, rank_data


class FaceRecognizer:
    def __init__(self, mode='tensorflow'):
        """
        Initalizes class containing functions & config for face recognition using pre-trained VGGFace.
        Modified from Chris' original code.
        ROC code adapted from: https://github.com/davidsandberg/facenet
        """
        if mode == 'torch':
            self.f_model = VggFace(mode='recognition')
            self.torch_tsf = transforms.ToTensor()
        elif mode == 'tensorflow':
            if not keras_vggface_available:
                raise RuntimeError('Keras VGGFace not available - please install!')

            from keras_vggface.vggface import VGGFace
            from keras_vggface import utils
            from keras.engine import Model

            self.utils = utils

            self.model = VGGFace(include_top=True)
            # Get 4096-D feature vector (layer prior to FC layer used for classification (face IDs))
            last_layer = self.model.get_layer('fc7/relu').output
            self.f_model = Model(self.model.input, last_layer)

        self.mode = mode
        self.features = 4096

    def prepare_im(self, image):
        """
        Prepares input image for VGGFace.
        :param image: PIL image.
        :return: Processed image in numpy format (1xHxWxC).
        """
        im = image.resize((224, 224))

        if self.mode == 'torch':
            im = self.torch_tsf(im)*255
            im = self.f_model.preprocess(im).unsqueeze(0)
        else:
            im = np.array(im).astype(np.float32)
            im = np.expand_dims(im, axis=0)
            im = self.utils.preprocess_input(im, version=1)

        return im

    @staticmethod
    def distance_feats(v, u, method='l2'):
        """
        Computes distance between two feature arrays using specified method.
        :param v: Input array 1 (NxF).
        :param u: Input array 2 (MxF).
        :param method: Method to use to compute distance between arrays.  Must be one of [
        'cityblock', 'cosine', 'euclidean', 'haversine', 'l2', 'l1', 'manhattan', 'precomputed', 'nan_euclidean']
        :return: Distance result (NxM).
        """
        method = method.lower()
        if method in distance_metrics():
            return distance_metrics()[method](v, u)
        else:
            raise ValueError('Distance method must be one of the following:', distance_metrics().keys())

    def extract_full_dir(self, in_dir, out_dir=None, thresholds=None, id_source=None, batch_size=16,
                         file_name='vgg_gallery', seed=0):  # Out of date, needs updating
        """
        Extracts all VGG features from given image directory.
        :param in_dir: Directory containing input images.
        :param out_dir: Directory to save output images.
        :param thresholds: tuple containing specific start/end positions from list of images.
        :param id_source: File containing corresponding IDs for given images.
        :param batch_size: Batch size to use when converting images.
        :param seed: Random seed to use for random generator.
        :return: Full feature set (Nx4096) gathered from input images.
        """
        # file setup
        files = [name for name in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, name))] # -> doesn't seem to work on the server... next(os.walk(in_dir))[2]
        files.sort()
        if thresholds is not None:
            files = files[thresholds[0]:thresholds[1]]
        random.seed(seed)

        raise RuntimeError('blacklist location needs to be redefined.')
        blacklist_loc = None

        blacklist = pd.read_csv(blacklist_loc, header=[0])['Images'].tolist()
        blacklist = [b.split('.')[0] for b in blacklist]
        blacklisted_indices = [not any(b in file for b in blacklist) for file in files]
        files = list(itertools.compress(files, blacklisted_indices))

        # Collection and linking of IDs with images
        file_id_link = {}

        if id_source is not None:
            with open(id_source, "r") as f:
                for index, line in enumerate(f):
                    file_id_link[line.split(' ')[0].split('.')[0]] = int(line.split(' ')[1])

        id_set = defaultdict(list)

        for index, file in enumerate(files):
            id = file_id_link[file.split('.')[0]]
            id_set[id].append(index)

        id_stack = np.zeros((len(id_set.keys()), ), dtype=np.int)
        gallery_files = []

        for index, (key, val) in enumerate(id_set.items()):
            keep_gallery = random.choice(val)
            gallery_files.append(files[keep_gallery])
            id_stack[index] = key

        # Batching, preparing images and running feature extraction
        out_stack = np.zeros((len(gallery_files), self.features))
        batch_available = False

        for index, file in enumerate(tqdm(gallery_files)):
            file_loc = os.path.join(in_dir, file)
            if index % batch_size == 0:
                im_stack = self.prepare_im(Image.open(file_loc))
                batch_available = True
                s_index = index
            else:
                im_stack = np.concatenate((im_stack, self.prepare_im(Image.open(file_loc))), axis=0)

            if im_stack.shape[0] == batch_size:  # When batch full, extract features
                out_stack[s_index:index+1, :] = self.f_model.predict(im_stack)
                batch_available = False

        if batch_available:  # catch any images left behind
            out_stack[s_index:index+1, :] = self.f_model.predict(im_stack)

        if out_dir is not None:  # Save to file as an .npz file
            np.savez(os.path.join(out_dir, file_name), out_stack=out_stack, id_stack=id_stack, file_stack=np.asarray(gallery_files))

        return out_stack, id_stack

    def extract_features_from_file(self, image):
        """
        Given image file location, opens image and extracts VGG features.
        :param image: Image file location (string).
        :return: Extracted features.
        """
        im = Image.open(image)
        im = self.prepare_im(im)
        if self.mode == 'torch':
            out = self.f_model(im).numpy()
        else:
            out = self.f_model.predict(im)
        return out

    def extract_features_from_array(self, image):
        """
        Converts image array into required format, then extracts VGG features.
        :param image: Image in numpy array format, (CxHxW or HxWxC).  Image must only contain positive numbers.
        :return: Extracted features.
        """
        if image.shape[0] == 3:
            im_np = image.transpose([1, 2, 0])
        else:
            im_np = image

        im = Image.fromarray(im_np.astype(np.uint8))
        im = self.prepare_im(im)
        if self.mode == 'torch':
            out = self.f_model(im).numpy()
        else:
            out = self.f_model.predict(im)

        return out

    def extract_features_from_batch(self, batch, format='ycbcr', im_type='jpg', max_val=1):

        batch_modif = np.copy(batch)
        if format == 'ycbcr':
            for i in range(batch_modif.shape[0]):
                batch_modif[i, ...] = np.clip(
                    ycbcr_convert(batch_modif[i, ...],
                                  input='ycbcr', max_val=max_val, im_type=im_type)*(255/max_val), 0, 255)

        if batch_modif.shape[1] == 3:
            batch_np = batch_modif.transpose([0, 2, 3, 1])
        else:
            batch_np = batch_modif

        if self.mode == 'torch':

            batched_input = torch.zeros([batch_np.shape[0],  batch_np.shape[-1], 224, 224])

            for i in range(batch_np.shape[0]):
                im = Image.fromarray(batch_np[i, ...].astype(np.uint8))
                batched_input[i, ...] = self.prepare_im(im)

            out = self.f_model(batched_input)
            return out.numpy()

        else:
            batched_input = np.zeros((batch_np.shape[0], 224, 224, batch_np.shape[-1]))
            for i in range(batch_np.shape[0]):
                im = Image.fromarray(batch_np[i, ...].astype(np.uint8))
                batched_input[i, ...] = self.prepare_im(im)

            out = self.f_model.predict(batched_input)
            return out

    @staticmethod
    def cumulative_match(probe_gallery_measure, probe_ids, gallery_ids, verbose=False, mode='dist',
                               resolve_ties=True, tie_mode='average', quick_probe=False, plot=False):
        """
        Function which traces the rank retrieval rate and produces a cumulative match curve
        for provided probe/gallery comparisons.
        :param probe_gallery_measure: Matrix of size NxM comparing each probe with each gallery feature set.
        :param probe_ids: Set of probe IDs (list of size N).
        :param gallery_ids: Set of gallery IDs (list of size M).
        :param verbose: Set to true to output diagnostic values.
        :param mode: Distance or Similarity measures.
        :param resolve_ties: Set to true to resolve ties.
        :param tie_mode: Method to use to break ties in distance/similarity measurements.
        :param quick_probe: Set to true to skip all aggregation methods and plotting and only return id_rank2.
        :param plot: Set to true to produce a CMC curve plot.
        :return: CMC_x=rank values, CMC_y=no. of subjects @ each rank (percentage), id_rank2=rank of each person/image/feature in probe
        """

        mode_types = ['sim', 'dist']
        if not(mode in mode_types):
            raise Exception("Incorrect mode; should be either 'dist' (for distance) or 'sim' (for similarity)")

        tie_mode_types = ['optimistic', 'pessimistic', 'average']
        if not(tie_mode in tie_mode_types):
            raise Exception("Incorrect tie mode; should be either 'optimistic', 'pessimistic', or 'average'")

        n_id = np.shape(probe_gallery_measure)[0]  # No. of probe subjects
        id_rank2 = np.ndarray(n_id)

        if verbose:
            print('There are', n_id, 'subjects/images/features in the probe set and', len(gallery_ids),
                  'subjects/images/features in the gallery set')

        for ctr, person_id in enumerate(probe_ids):
            if mode == 'sim':
                out1_sorted_idxs = np.flip(np.argsort(probe_gallery_measure[ctr]))
            elif mode == 'dist':
                out1_sorted_idxs = np.argsort(probe_gallery_measure[ctr])

            gallery_ids_sorted = [gallery_ids[x] for x in out1_sorted_idxs]

            rank = np.where(np.array(gallery_ids_sorted) == person_id)[0][0] + 1

            # Resolve any ties
            if resolve_ties:
                out1_sorted = [probe_gallery_measure[ctr][x] for x in out1_sorted_idxs]

                # Find how many scores have equal values for given subject
                score_idx = np.where(np.array(out1_sorted) == out1_sorted[rank-1])[0]

                gallery_ids_same_score = [gallery_ids_sorted[x] for x in score_idx]

                n_subjs_same = np.sum(np.asarray(gallery_ids_same_score) == person_id)
                n_subjs_unique = len(np.unique(np.asarray(gallery_ids_same_score)))

                if verbose:
                    print('')
                    print(score_idx)
                    print(person_id, gallery_ids_same_score)
                    print('Number of gallery subjects having same score of the current probe person id:', n_subjs_same)

                if len(score_idx) > 1 & n_subjs_unique > 1:

                    if verbose:
                        print('Old rank:', rank)
                    # The highest/worst rank is calculated as follows:
                    # the best possible rank + number of unique subjects - 1.
                    # For example, if probe feat subj id = 1 and gallery feats
                    # with subj ids [3,1,1] have an identical score to each other
                    # with possible ranks 2-4, then worst rank = best possible
                    # rank (2) + no. of unique subjects (2) - 1 = 3 - i.e. possible
                    # ranks now 2-3 instead of 2-4 to cater for probe subject
                    # appearing more than once; however, if probe subj id=3,
                    # then possible ranks remain 2-4
                    if tie_mode == 'optimistic':  # New rank (smallest (best) rank)
                        rank = score_idx[0] + 1
                    elif tie_mode == 'pessimistic':  # New rank (highest (worst) rank)
                        if n_subjs_same == 1:
                            rank = score_idx[-1] + 1
                        else:
                            rank = score_idx[0] + n_subjs_unique-1 + 1
                    elif tie_mode == 'average':  # New rank (average of optimistic and pessimistic (best and worst) ranks)
                        if n_subjs_same == 1:
                            rank = ((score_idx[0] + 1) + (score_idx[-1] + 1))/2.0
                        else:
                            rank = ((score_idx[0] + 1) + (score_idx[0] + n_subjs_unique-1 + 1))/2.0

                    if verbose:
                        print('New rank:', rank)

            id_rank2[ctr] = rank

        if verbose:
            print('')
            print(id_rank2)  # Rank of each ID
            print('')
            for ctr, x in enumerate(id_rank2):
                print('Person no.', ctr, 'with person ID', probe_ids[ctr], 'retrieved at Rank', x)
            print('')

        if quick_probe:
            return id_rank2

        CMC_x = list(range(1, len(gallery_ids)+1))

        CMC_y = [(np.sum(id_rank2 <= r)/n_id)*100.0 for r in range(1, len(gallery_ids)+1)]

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            plt.plot(CMC_x, CMC_y, '-o')
            plt.xlabel('Rank')
            plt.ylabel('Rank retrieval rate (%)')
            plt.grid(which='major')
            plt.title('Cumulative Match Curve (CMC)')

            if len(gallery_ids) > 10:
                x_ticks = np.arange(len(gallery_ids)+1, step=10)
                x_ticks[0] = 1
                plt.xticks(x_ticks)
            plt.rcParams["figure.figsize"] = (50, 10)
            plt.show()
            # TODO: add in to verbose system
            ## https://stackoverflow.com/questions/22272081/label-python-data-points-on-plot
            #for xy in zip(CMC_x, CMC_y):                                       # <--
            #    ax.annotate("(%s, %s)" % xy, xy=xy, textcoords='data') # <--

        return CMC_x, CMC_y, id_rank2

    @staticmethod
    def calculate_accuracy(threshold, dist, actual_issame, mode='dist'):
        """
        Calculates True positive and False positive rates given a distance measure and threshold.
        :param threshold: Positive/negative result threshold.
        :param dist: Distance/similarity between probe and gallery IDs.
        :param actual_issame: Expected results.
        :param mode: Distance or Similarity.
        :return: True Positive Rate, False Positive Rate, Accuracy.
        """
        if mode == 'dist':
            predict_issame = np.less(dist, threshold)  # If 'dist' contains distances (lower is better)
        elif mode == 'sim':
            predict_issame = np.greater(dist, threshold)  # If 'dist' contains similarity measures (higher is better)
        else:
            raise Exception("Incorrect mode; should be either 'dist' (for distance) or 'sim' (for similarity)")

        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

        tpr = 0 if (tp+fn == 0) else float(tp) / float(tp+fn)
        fpr = 0 if (fp+tn == 0) else float(fp) / float(fp+tn)
        acc = float(tp+tn)/dist.size
        return tpr, fpr, acc

    @staticmethod
    def ROC_calc(dist, actual_issame=None, mode='dist', verbose=False, thresh_min=0, thresh_max=1.01,
                 thresh_step=0.01, plot=False):
        """
        Function which calculates Receiver Operating Characteristics for given distance matrix.
        :param dist: Distance matrix (NxM).
        :param actual_issame: True matches corresponding to distance matrix.
        :param mode: Distance or similarity mode.
        :param verbose: Set to true to print diagnostic statements throughout code.
        :param thresh_min: Minimum threshold to apply for ROC calculation.
        :param thresh_max: Maximum threshold to apply for ROC calculation.
        :param thresh_step: Step to increment between minimum and maximum thresholds.
        :param plot: Set to true to plot the ROC characteristics.
        :return: False positive rate per threshold, True positive rate per threshold, Corresponding thresholds
        """

        n_faces = np.shape(dist)[0]  # No. of probe subjects
        n_id = np.shape(dist)[1]  # No. of gallery subjects

        if actual_issame is None:  # Assumes probe IDs are in the same sequence as the gallery IDs
            actual_issame = np.eye(N=n_faces, M=n_id).astype(bool)

        if verbose:
            print(n_faces, n_id)
            print(actual_issame)

        nrof_folds = n_faces
        thresholds = np.arange(thresh_min, thresh_max, thresh_step)
        nrof_thresholds = len(thresholds)
        tprs = np.zeros((nrof_folds, nrof_thresholds))
        fprs = np.zeros((nrof_folds, nrof_thresholds))

        # Loops through thresholds and calculate TPR, FPR for each threshold
        for person_id in range(n_faces):
            for threshold_idx, threshold in enumerate(thresholds):
                tprs[person_id, threshold_idx], fprs[person_id, threshold_idx], _ = \
                    FaceRecognizer.calculate_accuracy(threshold, dist[person_id], actual_issame[person_id], mode)

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)

        if verbose:
            print(thresholds)
            print(fpr)
            print(tpr)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            plt.plot(fpr, tpr, '-o')
            plt.xlabel('False Accept Rate (FAR)')
            plt.ylabel('True Accept Rate (TAR)')
            plt.grid(which='minor')
            plt.title('Receiver Operating Characteristics (ROC) curve')

            plt.semilogx()
            plt.rcParams["figure.figsize"] = (50, 10)

            ## https://stackoverflow.com/questions/22272081/label-python-data-points-on-plot
            #for xy in zip(CMC_x, CMC_y):                                       # <--
            #    ax.annotate("(%s, %s)" % xy, xy=xy, textcoords='data') # <--
            plt.show()

        return fpr, tpr, thresholds

    @staticmethod
    def ROC_main(dist, actual_issame, score_mode='dist', verbose=False, thresh_min=0, thresh_max=1.01,
                 thresh_step=0.01, plot=False):
        """
        Function which calculates Receiver Operating Characteristics and other metrics for given distance matrix.
        :param dist: Distance matrix (NxM).
        :param actual_issame: True matches corresponding to distance matrix.
        :param score_mode: Distance or similarity mode.
        :param verbose: Set to true to print diagnostic statements throughout code.
        :param thresh_min: Minimum threshold to apply for ROC calculation.
        :param thresh_max: Maximum threshold to apply for ROC calculation.
        :param thresh_step: Step to increment between minimum and maximum thresholds.
        :param plot: Set to true to plot the ROC characteristics.
        :return: Area-under-curve, Equal error rate, True positive rate, False positive rate, corresponding thresholds
        """
        if (score_mode != 'dist') & (score_mode != 'sim'):
            raise Exception("Incorrect mode; should be either 'dist' (for distance) or 'sim' (for similarity)")

        fpr, tpr, thresholds = FaceRecognizer.ROC_calc(dist, mode=score_mode, verbose=verbose,
                                                       actual_issame=actual_issame, thresh_min=thresh_min,
                                                       thresh_max=thresh_max, thresh_step=thresh_step, plot=plot)
        # Input must be a distance measure, where lower scores are better (i.e. not similarity)

        auc = metrics.auc(fpr, tpr)
        # TODO: confirm changing the interpolation can work
        eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0, 1)
        if verbose:
            print('Area Under Curve (AUC): %1.3f' % auc)
            print('Equal Error Rate (EER): %1.3f' % eer)

        if verbose:
            # To check: At EER, FAR and TAR should be identical
            t = eer
            t_idx = np.where(thresholds == t)[0]

            if len(t_idx) == 0:
                t = thresholds-t
                t_idx = np.sort(np.argsort(abs(t))[0:2])
                # Get 2 closest thresholds e.g. if t=0.444, get 0.44 and 0.45 (s.t. t lies between these thresholds)

            for idx in t_idx:
                print('At threshold='+str(thresholds[idx]), 'FAR='+str(fpr[idx]), 'and TAR='+str(tpr[idx]))

        return auc, eer, fpr, tpr, thresholds

    def full_package(self, fr_package, gallery, probes, gallery_ids, rank_step=5, rank_disp_limit=300,
                     thresh_step=10, verbose=False, query_name=None):

        metrics_package = defaultdict(list)
        CMC_package = defaultdict(list)

        if query_name is None:
            query_name = 'Unnamed'
        if verbose:
            print('FR rank retrievals:')

        dist = self.distance_feats(fr_package, gallery)

        CMC_x, CMC_y, ranks = self.cumulative_match(dist, mode='dist', verbose=False, probe_ids=probes,
                                                    gallery_ids=gallery_ids, plot=False)

        CMC_package[query_name + '_x'] = CMC_x
        CMC_package[query_name + '_y'] = CMC_y

        if verbose:
            if query_name is None:
                print('Results:')
            else:
                print(query_name)
            print(ranks)
            print('-------')

        actual_issame = get_actual_issame(probes, gallery_ids)

        auc, eer, fpr, tpr, thresholds = self.ROC_main(dist, score_mode='dist', verbose=False,
                                                       actual_issame=actual_issame, thresh_min=-thresh_step,
                                                       thresh_max=np.max(dist)+thresh_step,
                                                       thresh_step=thresh_step, plot=False)

        metrics_package['Key'].append(query_name)
        metrics_package['AUC'].append(auc)
        metrics_package['EER'].append(eer)

        for rank, percentage in zip(CMC_x, CMC_y):
            if rank == rank_disp_limit:
                metrics_package['Rank_' + str(rank)].append(percentage)
                break
            if rank % rank_step == 0:
                metrics_package['Rank_' + str(rank)].append(percentage)

        return CMC_package, metrics_package, ranks


def getColor(c, N, idx):
    cmap = mpl.cm.get_cmap(c)
    # cmap = cc.m_glasbey_dark
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))


def plot_cmc(cmc_data, save_loc='.', xlim=(75, 100), ylim=(45, 60)):

    cmc_copy = cmc_data.copy()
    cmc_copy.columns = pd.MultiIndex.from_tuples([tuple(reversed(c.split('-'))) for c in cmc_copy.columns])
    fig = plt.figure()
    n_colors = len(cmc_copy.columns.levels[0])
    for index, key in enumerate(cmc_copy.columns.levels[0]):
        plt.plot(cmc_copy.index, cmc_copy[key].mean(axis=1), label=key, color=getColor('nipy_spectral', n_colors, index))
        plt.xlabel('Rank')
        plt.ylabel('Rank retrieval rate (%)')
        plt.title('Cumulative Match Curve (CMC)')

    plt.grid(which='major')
    plt.legend()
    plt.savefig(os.path.join(save_loc, 'cmc_curves.pdf'))
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.savefig(os.path.join(save_loc, 'cmc_curves_zoomed.pdf'))
    plt.show()


# TODO: add a function just for loading metrics, which couples into same display functions used by eval system
# TODO: trim down the useless functions in this area
