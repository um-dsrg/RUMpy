from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
import os

from rumpy.sr_tools.data_handler import SuperResImages, CelebaSplitSampler, ClassifierImages, VideoSequenceImages


class StaggeredData:
    """
    CLASS IS UNFINISHED
    """

    def __init__(self, lowqp_handler=None, highqp_handler=None, high_counts=3):
        self.low_loader = lowqp_handler
        self.high_loader = highqp_handler
        self.low_handler = iter(lowqp_handler)
        self.high_handler = iter(highqp_handler)
        self.low_next = False
        self.low_len = len(lowqp_handler)
        self.high_len = len(highqp_handler)
        self.length = len(lowqp_handler) + len(highqp_handler)
        self.iters = 0
        self.low_iters = 0
        self.high_iters = 0
        self.high_counts = high_counts

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.iters == self.length:
            self.iters = 0
            self.low_iters = 0
            self.high_iters = 0
            self.low_handler = iter(self.low_loader)
            self.high_handler = iter(self.high_loader)
            raise StopIteration

        high_pass = True
        if self.iters != 0 and self.high_counts != 0:
            if (self.iters % self.high_counts == 0) and (self.low_iters != self.low_len):
                high_pass = False

        if high_pass and self.high_iters != self.high_len:
            lr_im, hr_im, lr_filenames, mask_im, halfway_im = next(self.high_handler)
            self.high_iters += 1
        else:
            lr_im, hr_im, lr_filenames, mask_im, halfway_im = next(self.low_handler)
            self.low_iters += 1

        self.iters += 1

        return lr_im, hr_im, lr_filenames, mask_im, halfway_im


# TODO: prepare a better dictionary system for data parameters
# TODO: this shouldn't be called SISR since it's a general function
def sisr_data_setup(training_sets, eval_sets, batch_size=16, eval_batch_size=1, dataloader_threads=8,
                    drop_last_training_batch=False, extract_masks=False, rep_partition=None, attributes=None,
                    multi_frame_config=None,
                    online_pipeline=None, blacklists=None, sampler_attributes=None, task_type='SR', **kwargs):
    """
    Prepares super-res data for training/eval with several custom parameters available.
    :param training_sets: training set parameter dictionaries
    :param eval_sets: eval set parameter dictionaries
    :param batch_size:  batch size for parallel loading of images
    :param eval_batch_size: batch size for evaluation images (default to 1 as images typically have different dimensions)
    :param dataloader_threads: number of threads to use for parallel data loading
    :param extract_masks: set to true to also extract face masks from HR locations
    :param rep_partition: currently unused
    :param attributes: Attributes for specified datasets (dict) e.g. facial features for celeba
    :param online_pipeline: Online degradation pipeline and any additional configuration options
    :param blacklists: Blacklists (images to skip) for each particular dataset (dict)
    :param drop_last_training_batch: Set to true to drop last batch if total amount of data not divisible by batch size
    :param kwargs: Any other parameters which are common to all datasets (e.g. model scale)
    :param sampler_attributes: All parameters for a custom data sampler.
    :param task_type: Task being evaluation (currently either SR or Classification)
    :return: Training/Eval data loaders
    """

    def setup_data(data_set, split):
        """
        This function is run for each dataset, and makes all the necessary preparations.
        :param data_set: Dataset parameters.
        :param split: Train/Eval/Test split
        :return: Dataset class
        """
        if extract_masks:
            mask_loc = os.path.join(data_set['hr'], 'segmentation_patterns')
        else:
            mask_loc = None

        custom_range = None

        if data_set['cutoff'] is not None:  # cutoff either specifies start/stop position, or just the stop position
            if type(data_set['cutoff']) == list:
                custom_range = data_set['cutoff']
            else:
                custom_range = (0, data_set['cutoff'])
        elif data_set['name'] is None:  # if not a particular dataset, take in all images
            split = 'all'

        if data_set['qpi_values'] is not None:  # catering for legacy code
            data_set['degradation_metadata'] = data_set['qpi_values']

        if data_set['degradation_metadata'] == 'on_site':  # degradation file should always have the same name
            data_set['degradation_metadata'] = os.path.join(data_set['lr'], 'degradation_metadata.csv')
            if not os.path.isfile(data_set['degradation_metadata']):  # catering for legacy code
                data_set['degradation_metadata'] = os.path.join(data_set['lr'], 'qpi_slices.csv')

        if blacklists is not None and data_set['name'] in blacklists:  # only relevant if blacklist provided
            blacklist = blacklists[data_set['name']]
        else:
            blacklist = None

        if attributes is not None and data_set['name'] is not None:
            data_attributes = attributes[data_set['name']]
        else:
            data_attributes = None

        # TODO: can these options be condensed to prevent the need for all these specifications?
        data_arguments = {'lr_dir': data_set['lr'],
                          'hr_dir': data_set['hr'],
                          'blacklist': blacklist,
                          'data_attributes': data_attributes,
                          'image_shortlist': data_set['image_shortlist'],
                          'metadata': data_set['metadata'],
                          'attribute_amplification': data_set['attribute_amplification'],
                          'dataset': data_set['name'],
                          'split': split, 'y_only': False if split == 'eval' else True,
                          'custom_split': custom_range,
                          'degradation_metadata_file': data_set['degradation_metadata'],
                          'legacy_blur_kernels': data_set['legacy_blur_kernels'],
                          'random_crop': data_set['crop'],
                          'random_augments': data_set['random_augment'],
                          'use_hflip': data_set['use_hflip'] if data_set['use_hlip'] is not None else True,
                          'use_vflip': data_set['use_vflip'] if data_set['use_vflip'] is not None else True,
                          'use_rotation': data_set['use_rotation'] or data_set['random_rot']if (data_set['use_rotation'] is not None and data_set['random_rot'] is not None) else True,
                          'use_random_colour_distort': data_set['use_random_colour_distort'],
                          'colour_distortion_strength': data_set['colour_distortion_strength'],
                          'recursive_search':
                              data_set['recursive_search'] if data_set['recursive_search'] is not None else False,
                          'mask_data': mask_loc,
                          'group_select': data_set['group_select'],
                          'online_degradations': data_set['online_degradations'],
                          'request_crops': data_set['request_crops'],
                          'augmentation_normalization': data_set['augmentation_normalization'],
                          'ignore_degradation_location': data_set['ignore_degradation_location'],
                          'online_degradation_params': online_pipeline,
                          'patch_selection_type': data_set['patch_selection_type'] if data_set['patch_selection_type'] is not None else 'random',
                          'attribute_skip': data_set['attribute_skip']}

        if multi_frame_config is not None:
            data_class = VideoSequenceImages(**data_arguments,
                                             **multi_frame_config,
                                             **kwargs)
        elif task_type == 'classification':
            data_arguments['predefined_patch_location'] = data_set['predefined_patch_location']
            data_class = ClassifierImages(**data_arguments, **kwargs)
        else:
            data_class = SuperResImages(**data_arguments, **kwargs)

        return data_class

    all_train_data = []
    all_val_data = []
    print('---------------')
    print('preparing training data:')
    for key, train_set in training_sets.items():
        all_train_data.append(setup_data(train_set, split='train'))
    print('---------------')
    print('preparing validation data:')
    for key, eval_set in eval_sets.items():
        all_val_data.append(setup_data(eval_set, split='eval'))
    print('---------------')

    if len(all_train_data) == 1:  # concatenates data if multiple datasets provided.
        all_train_data = all_train_data[0]
    else:
        all_train_data = ConcatDataset(all_train_data)

    if len(all_val_data) == 1:
        all_val_data = all_val_data[0]
    else:
        all_val_data = ConcatDataset(all_val_data)

    if sampler_attributes is None:
        sampler = None
    elif sampler_attributes['name'].lower() == 'celebasplitsampler':
        sampler = CelebaSplitSampler(all_train_data, **sampler_attributes)
    else:
        raise RuntimeError('Selected data sampler not recognized.')

    train_dataloader = DataLoader(dataset=all_train_data,
                                  batch_size=batch_size,
                                  shuffle=True if sampler is None else False,
                                  num_workers=dataloader_threads,
                                  pin_memory=True,
                                  drop_last=drop_last_training_batch,
                                  sampler=sampler)

    val_dataloader = DataLoader(dataset=all_val_data, batch_size=eval_batch_size)

    return train_dataloader, val_dataloader


# QPI SPLIT ZONE - TODO: if required again, need to fix and reinstate this section:
# high_counts = 0
# qpi_split = False
# if qpi_split:  # TODO: recheck here, things have changed since this implementation
#     rep_partition = ['q0']
#     train_data_aux = SuperResImages(lr_location, hr_location, dataset='celeba', split='train', custom_split=custom_train,
#                                     scale=scale, request_type=colorspace,
#                                     lr_type=input, mask_data=mask_loc,
#                                     group_select=rep_partition, halfway_data=HR_halfway_loc)
#     train_data_aux = DataLoader(dataset=train_data_aux,
#                                 batch_size=batch_size,
#                                 shuffle=True,
#                                 num_workers=dataloader_threads,
#                                 pin_memory=True,
#                                 drop_last=False)
#
#     rep_partition = ['q1', 'q2', 'q3']
# if qpi_split:
#     train_dataloader = StaggeredData(lowqp_handler=train_data_aux, highqp_handler=train_data, high_counts=high_counts)

# if specific_cut is True:
#     sampler = ProgressiveQPISampler(train_data, partitions=partitions, specific_cut=specific_cut)
# else:
#     sampler = None
