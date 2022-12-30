import random

from rumpy.image_tools.image_manipulation.image_functions import downsample, upsample


class Downsample:
    """
    Downsampling system to reduce the size of the image.
    """

    def __init__(self, scale=4, jm=False, random_scale=False, scale_range=(2, 8),
                 normalize_metadata=True, restrict_metadata=False):
        """
        Downsampler for the image manipulation system.

        :param scale: Scaling factor used to downsample the image.
        :param jm: Set this flag when JM compression is used (applies certain constraints).
        :param random_scale: Set this flag to generate a random scaling factor per image.
        :param scale_range: Tuple containing the minimum and maximum scaling factors.
        :param normalize_metadata: Set this flag to normalize the scaling factor using the range.
        :param restrict_metadata: Set this flag to restrict the system from outputting metadata.
        """

        self.scale = scale
        self.random_scale = random_scale
        self.scale_range = scale_range
        self.jm = jm
        self.normalize_metadata = normalize_metadata
        self.restrict_metadata = restrict_metadata

    def get_hyperparams(self):
        """
        Get extra information on the downsampling parameters used.
        """
        return {'min_scale': self.scale_range[0], 'max_scale': self.scale_range[1]}

    def normalize(self, scale):
        """
        Normalize the given scale factor.

        :param scale: Scaling factor used to downsample the image.
        :returns: The normalized scale factor, depending on the scale range.
        """
        return (scale - self.scale_range[0]) / (self.scale_range[1] - self.scale_range[0])

    def __call__(self, image):
        if self.random_scale:
            scale = random.randint(self.scale_range[0], self.scale_range[1])
        else:
            scale = self.scale

        _, lr_im = downsample(image, scale=scale, jm=self.jm)

        if self.normalize_metadata:
            scale = self.normalize(scale)

        if self.restrict_metadata:
            return lr_im, {}
        else:
            return lr_im, {'scale': scale}


class Upsample:
    """
    Upsampling system to increase the size of the image.
    """

    def __init__(self, scale=4, random_scale=False, scale_range=(2, 8), normalize_metadata=True):
        """
        Upsampler for the image manipulation pipeline.

        :param scale: Scaling factor used to upsample the image.
        :param random_scale: Set this flag to generate a random scaling factor per image.
        :param scale_range: Tuple containing the minimum and maximum scaling factors.
        :param normalize_metadata: Set this flag to normalize the scaling factor using the range.
        """
        self.scale = scale
        self.random_scale = random_scale
        self.scale_range = scale_range
        self.normalize_metadata = normalize_metadata

    def get_hyperparams(self):
        """
        Get extra information on the upsampling parameters used.
        """
        return {'min_scale': self.scale_range[0], 'max_scale': self.scale_range[1]}

    def normalize(self, scale):
        """
        Normalize the given scale factor.

        :param scale: Scaling factor used to upsample the image.
        :returns: The normalized scale factor, depending on the scale range.
        """
        return (scale - self.scale_range[0]) / (self.scale_range[1] - self.scale_range[0])

    def __call__(self, image):
        if self.random_scale:
            scale = random.randint(self.scale_range[0], self.scale_range[1])
        else:
            scale = self.scale

        _, hr_im = upsample(image, scale=scale)

        if self.normalize_metadata:
            scale = self.normalize(scale)

        return hr_im, {'scale': scale}

