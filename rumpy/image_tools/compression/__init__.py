import os
import shutil
import random
import numpy as np
import torch.utils.data as tdata
from datetime import datetime

from rumpy.shared_framework.configuration import constants as sconst
from rumpy.image_tools.compression.compression_utils import jm_compress, jpeg_compress, ffmpeg_compress
from rumpy.sr_tools.helper_functions import create_dir_if_empty


class JMCompress:
    """
    Compression system powered by the JM H.264 encoder-decoder.
    """
    def __init__(self, qpi=28, compression_range=(20, 40), random_compression=False, verbose=False,
                 temp_save_dir=sconst.scratch_directory, normalize_metadata=True):
        """
        JM compression module for the image manipulation system.

        :param qpi: Quantization parameter to control the amount of compression (lower value give better quality).
        :param compression_range: Tuple containing the minimum and maximum quantization parameter.
        :param random_compression: Set this flag to choose a random quantization parameter per image.
        :param verbose: Set this flag to print detailed information from the JM software.
        :param temp_save_dir: Location to store the temporary files from the JM software.
        :param normalize_metadata: Set this flag to normalize the quantization parameter using the range.
        """

        if qpi > 51 or compression_range[1] > 51:
            raise RuntimeError('QPI cannot be larger than 51.')

        specific_temp_save_dir = os.path.join(temp_save_dir, datetime.today().strftime("%Hh-%Mm-%Ss-%b-%d-%Y"))
        create_dir_if_empty(specific_temp_save_dir)

        self.jm_command, self.temp_files = self.jm_path_cmd_setup(specific_temp_save_dir)
        self.yuv_loc = self.temp_files[0]
        self.comp_loc = self.temp_files[1]
        self.verbose = verbose
        self.random_compression = random_compression
        self.qpi = qpi
        self.compression_range = compression_range
        self.normalize_metadata = normalize_metadata
        self.temp_save_dir = specific_temp_save_dir

    def get_hyperparams(self):
        """
        Get extra information on the compression parameters used.
        """
        return {'min_qpi': self.compression_range[0], 'max_qpi': self.compression_range[1]}

    def jm_path_cmd_setup(self, out_dir, temp_id=''):
        """
        Set the temporary files and create the command to run the JM software.

        :param out_dir: Location to output the temporary files.
        :param temp_id: Suffix to identify the temporary files in the case of multiple processes.

        :return: The full JM command and a list of the paths for the temporary files.
        """
        # creation of temporary locations for video transfer to JM and back
        temp_yuv_loc = os.path.join(out_dir, 'vid_temp_%s.yuv' % temp_id)
        temp_comp_loc = os.path.join(out_dir, 'vid_comp_%s.yuv' % temp_id)
        temp_h264_loc = os.path.join(out_dir, 'vid_comp_%s.h264' % temp_id)
        temp_stats_loc = os.path.join(out_dir, 'comp_stats_%s.dat' % temp_id)
        temp_leakybucket = os.path.join(out_dir, 'leakybucketparam.cfg')
        temp_data = os.path.join(os.getcwd(), 'data.txt')
        temp_log = os.path.join(os.getcwd(), 'log.dat')

        # Setting up constant JM params
        jm_params = {'InputFile': temp_yuv_loc, 'OutputFile': temp_h264_loc,
                     'ReconFile': temp_comp_loc, 'StatsFile': temp_stats_loc,
                     'LeakyBucketParamFile': temp_leakybucket,
                     'NumberBFrames': 0, 'IDRPeriod': 1, 'IntraPeriod': 1, 'QPISlice': 0,
                     'SourceHeight': 0, 'SourceWidth': 0, 'FramesToBeEncoded': 1}

        jm_bin = os.path.join(os.path.dirname(sconst.base_directory), 'JM/bin')

        jm_command = jm_bin + '/lencod.exe -d ' + jm_bin + '/encoder_baseline.cfg'

        for key, val in jm_params.items():
            jm_command += ' -p ' + str(key) + '=' + str(val)

        return jm_command, [temp_yuv_loc, temp_comp_loc, temp_h264_loc, temp_stats_loc, temp_leakybucket,
                            temp_data, temp_log]

    @staticmethod
    def cleanup(temp_files, verbose=False):
        """
        Cleans up temporary files produced by JM.
        """
        for location in temp_files:
            if os.path.exists(location):
                if os.path.isdir(location):
                    try:
                        shutil.rmtree(location)
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))
                else:
                    try:
                        os.remove(location)
                    except OSError as e:
                        print('%s not removed due to an error.' % location)
            else:
                if verbose:
                    print('%s does not exist.' % location)

    def normalize(self, qpi):
        """
        Normalize the given quantization parameter.

        :param qpi: Quantization parameter used to compress the image.
        :returns: The normalized quantization parameter, depending on the range.
        """
        return (qpi - self.compression_range[0]) / (self.compression_range[1] - self.compression_range[0])

    def compress(self, image, jm_command, temp_files):
        """
        Compress a single image using the given JM command.

        :param image: The image to be compressed.
        :param jm_command: The full command to run the JM software.
        :param temp_files: The list of locations for temporary files.

        :return: The compressed image together with the compression metadata.
        """
        if self.random_compression:
            qpi = random.randint(self.compression_range[0], self.compression_range[1])
        else:
            qpi = self.qpi

        output = jm_compress(image, qpi, jm_command, temp_files[0], temp_files[1], verbose=self.verbose)

        if self.normalize_metadata:
            qpi = self.normalize(qpi)

        return output, {'qpi': qpi}

    def call_with_tag(self, image, tag=''):
        jm_command, temp_files = self.jm_path_cmd_setup(self.temp_save_dir, temp_id=tag)
        output, metadata = self.compress(image, jm_command, temp_files)

        return output, metadata

    def __call__(self, image):
        output, metadata = self.compress(image, self.jm_command, self.temp_files)
        return output, metadata


class JPEGCompress:
    def __init__(self, quality=50, compression_range=(20, 80), random_compression=False, normalize_metadata=True):
        self.random_compression = random_compression
        self.quality = quality
        self.compression_range = compression_range
        self.normalize_metadata = normalize_metadata

    def get_hyperparams(self):
        return {'min_quality': self.compression_range[0], 'max_quality': self.compression_range[1]}

    def normalize(self, quality):
        return (quality - self.compression_range[0]) / (self.compression_range[1] - self.compression_range[0])

    def __call__(self, image):
        if self.random_compression:
            quality = random.randint(self.compression_range[0], self.compression_range[1])
        else:
            quality = self.quality

        output = jpeg_compress(image, quality)

        if self.normalize_metadata:
            quality = self.normalize(quality)

        return output, {'quality': quality}


class FFMPEGCompress:
    def __init__(self, qpi=28, compression_range=(20, 40), random_compression=False, verbose=False, normalize_metadata=True,
                 encoder_args=None, decoder_args=None, shift_encoder_qp=False, qp_shift_value=3):

        if qpi > 51 or compression_range[1] > 51:
            raise RuntimeError('QPI cannot be larger than 51.')

        self.verbose = verbose
        self.random_compression = random_compression
        self.qpi = qpi
        self.shift_encoder_qp = shift_encoder_qp
        self.qp_shift_value = qp_shift_value
        self.compression_range = compression_range
        self.normalize_metadata = normalize_metadata

        # These args try to match the parameters of the JM H.264 system
        # encoder_args = {
        #     'format': 'image2pipe',
        #     'vcodec': 'libx264',
        #     'preset': 'medium',
        #     'profile:v': 'baseline',
        #     # 'crf': qpi,
        #     'qp': qpi,
        #     'pix_fmt': 'yuv420p',
        #     'bf': 0,
        #     'r': 30,
        #     'b:v': 45020,
        #     'level': 4,
        #     'bsf:v': 'h264_mp4toannexb',
        #     # 'colorspace': 'bt709'
        # }

        if encoder_args:
            self.encoder_args = encoder_args
        else:
            self.encoder_args = {
                'format': 'image2pipe',
                'vcodec': 'libx264',
                'profile:v': 'baseline',
                'qcomp': 1.0,
                'pix_fmt': 'yuv420p',
            }

        if decoder_args:
            self.decoder_args = decoder_args
        else:
            self.decoder_args = {
                'format': 'rawvideo',
                'pix_fmt': 'rgb24',
            }

    def get_hyperparams(self):
        return {'min_qpi': self.compression_range[0], 'max_qpi': self.compression_range[1]}

    def normalize(self, qpi):
        return (qpi - self.compression_range[0]) / (self.compression_range[1] - self.compression_range[0])

    def compress(self, image):

        if self.random_compression:
            qpi = random.randint(self.compression_range[0], self.compression_range[1])
        else:
            qpi = self.qpi

        if self.shift_encoder_qp:
            # For libx264, the per-frame QPI seems to be 3 less than the 'video' QPI, so +3 needs to be added to compensate
            self.encoder_args['qp'] = qpi + self.qp_shift_value
        else:
            self.encoder_args['qp'] = qpi

        width, height = image.size
        self.decoder_args['s'] = '{}x{}'.format(width, height)

        output = ffmpeg_compress(image, self.encoder_args, self.decoder_args, verbose=self.verbose)

        if self.normalize_metadata:
            qpi = self.normalize(qpi)

        return output, {'qpi': qpi}

    def __call__(self, image):
        output, metadata = self.compress(image)
        return output, metadata


class RandomCompress:
    def __init__(self, jm_params, jpeg_params):
        self.jm_class = JMCompress(**jm_params)
        self.jpeg_class = JPEGCompress(**jpeg_params)

    def get_hyperparams(self):
        return {'min_jpeg_quality': self.jpeg_class.compression_range[0],
                'max_jpeg_quality': self.jpeg_class.compression_range[1],
                'min_qpi': self.jm_class.compression_range[0],
                'max_qpi': self.jm_class.compression_range[1]}

    def __call__(self, image):

        if np.random.uniform() < 0.5:
            if tdata.get_worker_info() is not None:  # helps prevent issues with overwriting when using multiple threads
                rand_id = tdata.get_worker_info().id
                processed_image, metadata = self.jm_class.call_with_tag(image, rand_id)
            else:
                processed_image, metadata = self.jm_class(image)

            metadata['jm_qpi'] = metadata.pop('qpi')
        else:
            processed_image, metadata = self.jpeg_class(image)
            metadata['jpeg_quality'] = metadata.pop('quality')

        metadata = {**{'jm_qpi': 0, 'jpeg_quality': 0}, **metadata}

        return processed_image, metadata
