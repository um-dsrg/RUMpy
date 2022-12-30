import os

# standard locations
base_directory = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
results_directory = os.path.join(os.path.dirname(base_directory), 'Results')
data_directory = os.path.join(os.path.dirname(base_directory), 'Data')
scratch_directory = os.path.join(os.path.dirname(base_directory), 'Scratch')
code_base_directory = os.path.join(os.path.dirname(base_directory), 'rumpy')
temp_dump = os.path.join(results_directory, 'temp')  # Temporary image dump

# Dataset Splits
data_splits = {'celeba': {'train': (0, 162770),
                          'eval': (162770, 182637),
                          'test': (182637, 202599)},
               'div2k': {'train': (0, 800),
                         'eval': (800, 900)},
               'flickr2k': {'train': (0, 2650)}}

# Other Configs
vggface_weights = os.path.join(os.path.dirname(base_directory), 'external_packages', 'VGGFace', 'vgg_face_dag.pth')
lightcnn_weights = os.path.join(os.path.dirname(base_directory), 'external_packages', 'LightCNN',
                                'LightCNN_29Layers_checkpoint.pth.tar')

temp_dump = os.path.join(results_directory, 'temp')  # Temporary image dump

metric_best_val = {
    'val-loss': 'lower',
    'train-loss': 'lower',
    'regression-loss': 'lower',
    'contrastive-loss': 'lower',
    'val-PSNR': 'higher',
    'val-SSIM': 'higher',
    'val-LPIPS': 'lower'
}


class TwoWayDict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


blur_kernel_code_conversion = TwoWayDict()
blur_kernel_code_conversion['iso'] = 0
blur_kernel_code_conversion['aniso'] = 1
blur_kernel_code_conversion['generalized_iso'] = 2
blur_kernel_code_conversion['generalized_aniso'] = 3
blur_kernel_code_conversion['plateau_iso'] = 4
blur_kernel_code_conversion['plateau_aniso'] = 5
blur_kernel_code_conversion['sinc'] = 6
