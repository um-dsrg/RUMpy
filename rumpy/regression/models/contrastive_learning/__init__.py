from rumpy.shared_framework.models.base_architecture import BaseModel
from rumpy.regression.models.contrastive_learning.encoding_models import Encoder, IDMN
import torchvision.models as models
import torch
import numpy as np


def noise_logic(noise_class, noise_colour, magnitude, magnitude_split=2, split_noise_mag=True):

    if split_noise_mag:
        logic_label = [0, 0, 0]
        logic_label[0] = partition_magnitude(magnitude, magnitude_split)
        colour_index = 1
        class_index = 2
    else:
        logic_label = [0, 0]
        colour_index = 0
        class_index = 1

    if noise_colour == 'gray':
        logic_label[colour_index] += 1
    if noise_class == 'gaussian':
        logic_label[class_index] += 1

    return logic_label


def compression_logic(compression_class, magnitude, magnitude_split=2, class_split=False):

    if class_split:
        logic_label = [0, 0]
        if 'jm' in compression_class:
            logic_label[1] = 1
        elif 'jpeg' in compression_class:
            logic_label[1] = 0
        else:
            raise RuntimeError('Unrecognized compression class.')
    else:
        logic_label = [0]

    logic_label[0] = partition_magnitude(magnitude, magnitude_split)

    return logic_label


def blur_logic(blur_class, sigma_x, sigma_y, magnitude_split=3):

    logic_label = [0, 0, 0]
    logic_label[0] = int(blur_class)
    logic_label[1] = partition_magnitude(sigma_x, magnitude_split)
    logic_label[2] = partition_magnitude(sigma_y, magnitude_split)
    return logic_label


def partition_magnitude(magnitude, splits=2):
    if splits == 2:
        if magnitude > 0.5:
            return 1
        else:
            return 0

    elif splits == 3:

        if magnitude > 0.66:
            return 2
        elif magnitude > 0.33:
            return 1
        else:
            return 0


# TODO: can all these functions be condensed somehow?
def register_metadata(keys):
    """
    Registers available metadata in a list of standard names.
    :param keys: Raw metadata keys.
    :return: Standardized metadata keys.
    """
    processed_keys = []
    for key in keys:
        if 'gaussian_noise' in key:
            processed_keys.append('gaussian_noise_scale')
        elif 'poisson_noise' in key:
            processed_keys.append('poisson_noise_scale')
        elif 'downsample' in key:
            processed_keys.append('scale')
        elif 'gray_noise' in key:
            processed_keys.append('gray_noise_boolean')
        elif 'jpeg' in key:
            processed_keys.append('jpeg_quality_factor')
        elif 'qpi' in key:
            processed_keys.append('jm_qpi')
        elif 'realesrganblur' in key:
            processed_keys.append(key.split('realesrganblur-')[-1])
        else:
            processed_keys.append('unknown')
    return processed_keys


def partition_metadata(metadata_mapping, selected_metadata='all', labelling_strategy='default'):
    """
    Defines which metadata is available, and calculates how class labels will be assigned.
    :param metadata_mapping: Mapping of data description strings to vector position.
    :param selected_metadata: Either 'all' (all available metadata) or a list of specific types.
    :param labelling_strategy: Defined in BaseContrastive docstring.
    :return: Available data types (list), magnitudes of each decision node (list), and maximum number
    of classes possible for data supplied (int).
    """

    if selected_metadata == 'all':
        accepted_data = ['blur', 'compression', 'noise']
    else:
        accepted_data = selected_metadata

    available_classes = []
    decisions = []

    if 'poisson_noise_scale' in metadata_mapping and 'noise' in accepted_data:
        available_classes.append('noise')
        if labelling_strategy == 'default':
            decisions.extend([2, 2])
        elif labelling_strategy == 'double_precision':
            decisions.extend([2, 2, 2])
        elif labelling_strategy == 'triple_precision':
            decisions.extend([3, 2, 2])

    if ('jpeg_quality_factor' in metadata_mapping or 'jm_qpi' in metadata_mapping) and 'compression' in accepted_data:

        available_classes.append('compression')
        if labelling_strategy == 'default' or labelling_strategy == 'double_precision':
            decisions.extend([2])
        elif labelling_strategy == 'triple_precision':
            decisions.extend([3])

        if 'jpeg_quality_factor' in metadata_mapping and 'jm_qpi' in metadata_mapping:
            decisions.append(2)  # indicator between the different compression types
            available_classes.append('jm_jpg_compression')

    if 'kernel_type' in metadata_mapping and 'blur' in accepted_data:
        available_classes.append('blur')
        decisions.extend([7, 3, 3])

    num_classes = np.prod(decisions)
    decision_mags = []
    for index, d in enumerate(decisions):
        if index == 0:
            decision_mags.append(1)
        else:
            decision_mags.append(np.prod(decisions[0:index]))

    return available_classes, decision_mags, num_classes


def degradation_vector_setup(available_classes):
    vector_size = 0
    for degradation in available_classes:
        if degradation == 'noise':
            vector_size += 2
        elif degradation == 'compression':
            vector_size += 2
        elif degradation == 'blur':
            vector_size += 2

    return vector_size


def vector_retrieval(metadata, valid_metadata, m_map):
    """
    Converts metadata into a unique class according to its composition.
    :param metadata: Metadata to classify.
    :param valid_metadata: Metadata which can be analyzed (list).
    :param m_map: Mapping of data description strings to vector position.
    :return: Assigned degradation vector.
    """

    vector = torch.zeros(degradation_vector_setup(valid_metadata))
    vec_pointer = 0

    if 'noise' in valid_metadata:

        if metadata[m_map['gaussian_noise_scale']] > 0:
            vector[vec_pointer] = metadata[m_map['gaussian_noise_scale']]
        else:
            vector[vec_pointer + 1] = metadata[m_map['poisson_noise_scale']]
        vec_pointer += 2

    if 'compression' in valid_metadata:

        if ('jpeg_quality_factor' in m_map and metadata[m_map['jpeg_quality_factor']] > 0) or 'jm_qpi' not in m_map:
            vector[vec_pointer] = metadata[m_map['jpeg_quality_factor']]
        else:
            vector[vec_pointer+1] = metadata[m_map['jm_qpi']]
        vec_pointer += 2

    if 'blur' in valid_metadata:
        vector[vec_pointer] = metadata[m_map['sigma_x']]
        vector[vec_pointer+1] = metadata[m_map['sigma_y']]

    return vector


def class_retrieval(metadata, valid_metadata, m_map, decision_mags, total_classes, labelling_strategy='default'):
    """
    Converts metadata into a unique class according to its composition.
    :param metadata: Metadata to classify.
    :param valid_metadata: Metadata which can be analyzed (list).
    :param m_map: Mapping of data description strings to vector position.
    :param decision_mags: Magnitude to apply to each decision tree branch.
    :param total_classes: The total number of possible classes (for error checking purposes).
    :param labelling_strategy: Defined in BaseContrastive docstring.
    :return: Assigned class (integer).
    """

    if labelling_strategy == 'double_precision':
        split = 2
        split_noise_magnitude = True
    elif labelling_strategy == 'triple_precision':
        split_noise_magnitude = True
        split = 3
    else:
        split = 2
        split_noise_magnitude = False

    decision_tree = []
    if 'noise' in valid_metadata:
        if metadata[m_map['gaussian_noise_scale']] > 0:
            n_class = 'gaussian'
            noise_mag = metadata[m_map['gaussian_noise_scale']]
        else:
            n_class = 'poisson'
            noise_mag = metadata[m_map['poisson_noise_scale']]

        if metadata[m_map['gray_noise_boolean']] > 0:
            n_colour = 'gray'
        else:
            n_colour = 'colour'

        decision_tree.extend(
            noise_logic(n_class, n_colour, noise_mag, magnitude_split=split, split_noise_mag=split_noise_magnitude))

    if 'compression' in valid_metadata:

        if ('jpeg_quality_factor' in m_map and metadata[m_map['jpeg_quality_factor']] > 0) or 'jm_qpi' not in m_map:
            c_class = 'jpeg'
            c_mag = metadata[m_map['jpeg_quality_factor']]
        else:
            c_class = 'jm'
            c_mag = metadata[m_map['jm_qpi']]

        if 'jm_jpg_compression' in valid_metadata:
            class_split = True
        else:
            class_split = False

        decision_tree.extend(compression_logic(c_class, c_mag, magnitude_split=split,
                                               class_split=class_split))

    if 'blur' in valid_metadata:
        blur_class = metadata[m_map['kernel_type']]
        sigma_x = metadata[m_map['sigma_x']]
        sigma_y = metadata[m_map['sigma_y']]

        decision_tree.extend(blur_logic(blur_class, sigma_x=sigma_x, sigma_y=sigma_y))

    current_label = 0
    for mag, d in zip(reversed(decision_mags), reversed(decision_tree)):  # runs through tree, starting from the source node
        if d != 0:
            current_label += mag + ((d-1)*mag)  # adds the branch magnitude + any additional weighting if more than two branches available at this node

    if current_label >= total_classes:
        raise RuntimeError('Label is greater than the total number of possible classes.')

    return current_label


class BaseContrastive(BaseModel):
    def __init__(self, device,
                 use_noise_injection=False,
                 noise_injection_frequency=0,
                 noise_injection_sigma=0.1,
                 labelling_strategy='default',
                 override_queue=False,
                 **kwargs):
        """
        Base structure used for all contrastive models.
        :param device: Device to run computations on (either CPU or any available GPUs)
        :param use_noise_injection: Set to true to inject noise into model during training.
        :param noise_injection_frequency: Sets frequency of noise injection (measured in epochs),
        :param noise_injection_sigma: Sigma (magnitude) of noise to inject.
        :param labelling_strategy: Type of class labelling strategy.  Options:
        'default' - Magnitude splits disregarded.
        'double_precision' -  All magnitude splits set to 2.
        'triple_precision' - All magnitude splits set to 3.
        :param override_queue: Set to true to re-initialize supmoco queue from scratch,
        even if a previous queue is available.
        :param kwargs: All other standard model arguments.
        """
        super(BaseContrastive, self).__init__(device=device, **kwargs)
        self.colorspace = 'rgb'
        self.im_input = 'unmodified'

        if labelling_strategy == 'half_precision':  # backward compatibility
            labelling_strategy = 'double_precision'

        self.labelling_strategy = labelling_strategy
        self.use_noise_injection = use_noise_injection
        self.noise_injection_frequency = noise_injection_frequency
        self.noise_injection_sigma = noise_injection_sigma
        self.eval_request_loss = False  # loss cannot be calculated on eval data

        self.training_metadata_mapping = {}
        self.valid_metadata = []
        self.decision_mags = []
        self.total_classes = 0
        self.regressor_type = 'contrastive'
        self.metadata_registered = False
        self.override_queue = override_queue
        self.degradation_vector_size = 0

    def register_training_metadata(self, metadata_keys):

        if not hasattr(self, 'data_type'):
            raise RuntimeError('Need to supply the degradation data types to analyze.')
        processed_keys = register_metadata(metadata_keys)
        self.training_metadata_mapping = {key: processed_keys.index(key) for key in processed_keys}
        self.valid_metadata, self.decision_mags, self.total_classes = partition_metadata(self.training_metadata_mapping,
                                                                                         self.data_type,
                                                                                         labelling_strategy=self.labelling_strategy)
        self.degradation_vector_size = degradation_vector_setup(self.valid_metadata)


    @staticmethod
    def define_encoder_model(model_name):
        """
        Defines the encoder to use with the contrastive system selected.
        :param model_name: String with model name (either matching pytorch defaults or other defined models)
        :return: model class for object instantiation
        """

        model_names = sorted(name for name in models.__dict__ if name.islower()
                             and not name.startswith("__") and callable(models.__dict__[name]))
        if model_name in model_names:
            model_class = models.__dict__[model_name]
        elif model_name == 'default':
            model_class = Encoder
        elif model_name.lower() == 'idmn':
            model_class = IDMN
        else:
            raise RuntimeError('Existing model name must be in PyTorch list of torchvision models.')
        return model_class

    def class_logic(self, metadata, keys):
        """
        Function which defines labels for artificially degraded images.
        :param metadata: List of metadata for image batch provided.
        :param keys: Keyword for each metadata value.
        :return: list of labels
        """

        if not self.metadata_registered:
            self.register_training_metadata([key[0] for key in keys])
            self.metadata_registered = True

            if self.__class__.__name__ == 'SupMoCoHandler':
                # the queue can be pre-populated if loading a model checkpoint
                if not hasattr(self.net, 'queue_labels') or self.override_queue or int(max(self.net.queue_labels)) >= self.total_classes:
                    self.net.register_classes(self.total_classes)  # resets the queue and updates class count
                else:
                    self.net.set_class_count(self.total_classes)  # only updates the class count

        labels = torch.zeros((1, metadata.size()[0])).to(device=self.device)
        m_map = self.training_metadata_mapping

        for index in range(labels.size()[1]):
            labels[0, index] = class_retrieval(metadata[index, :], self.valid_metadata, m_map, self.decision_mags,
                                               total_classes=self.total_classes, labelling_strategy=self.labelling_strategy)

        return labels

    def vector_logic(self, metadata, keys):

        if not self.metadata_registered:
            self.register_training_metadata([key[0] for key in keys])
            self.metadata_registered = True
            if not hasattr(self.net, 'queue_vectors') or self.override_queue or self.degradation_vector_size != self.net.queue_vectors.size()[0]:
                self.net.register_vector(self.degradation_vector_size)  # resets the queue and updates vector size

        vectors = torch.zeros((self.degradation_vector_size, metadata.size()[0]))
        m_map = self.training_metadata_mapping

        for index in range(vectors.size()[1]):
            vectors[:, index] = vector_retrieval(metadata[index, :], self.valid_metadata, m_map)

        return vectors

    def get_embedding_len(self):
        test_im = torch.zeros((1, 3, 10, 10)).to(self.device)
        self.net.eval()
        try:
            embedding = self.net.forward(test_im, test_im, get_q=True)[0]
        except:
            embedding = self.net.forward(test_im)[0]
        return embedding.shape[1]

    def run_model(self, x, *args, **kwargs):
        return self.net.forward(x, x, **kwargs)

    def add_gaussian_noise_to_model(self, sigma=0.1):
        with torch.no_grad():
            for param in self.net.parameters():
                param.add_(torch.randn(param.size(), device=self.device) * sigma)

    def epoch_end_calls(self):
        if self.use_noise_injection:
            if self.curr_epoch % self.noise_injection_frequency == 0:
                self.add_gaussian_noise_to_model(self.noise_injection_sigma)
