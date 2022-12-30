import os
import sys
import toml
import click
import click_config_file

import rumpy.shared_framework.configuration.constants as sconst


# config file loader
def toml_provider(file_path, cmd_name):
    return toml.load(file_path)


results_directory = os.path.join(os.path.dirname(sconst.base_directory), 'Results')
data_directory = os.path.join(os.path.dirname(sconst.base_directory), 'Data')


@click.command()
# Data Config
@click.option("--hr_dir", default=None, help='HR image directory.')
@click.option("--lr_dir", default=None, help='LR image directory.')
@click.option('--data_attributes', default=None,
              help='Additional data attributes (such as gender etc)')
@click.option("--batch_size", default=1, help='Batch size for parallel data loading.', show_default=True)
@click.option('--gallery_source', default=os.path.join(sconst.data_directory,
                                                       'celeba/png_samples/celeba_png_align/eval_vgg_features'),
              help='VGG Evaluation gallery directory.')
@click.option('--galleries', multiple=True, default=['gallery_0.npz', 'gallery_1.npz', 'gallery_2.npz',
                                                     'gallery_3.npz', 'gallery_4.npz'],
              help='Specific galleries to use for FR check.  If FR stats check not required, '
                   'will use first value in list.')
@click.option('--full_directory', is_flag=True, help='Set this flag to ignore any data partitions or splits.')
@click.option('--use_celeba_blacklist', is_flag=True,
              help='Set this flag to remove any images which form part of the blacklist.')
@click.option('--qpi_selection', type=(int, int),
              help='Set these values to enforce qpi range when selecting validation data.', default=(None, None))
@click.option('--gallery_ref_images', type=str, default=None,
              help='Location of gallery reference images, if different from HR directory.')
@click.option('--dataset_name', default=None, help='Specify dataset name to use associated eval split.')
@click.option('--group_select', multiple=True, default=None,
              help='Specify which group of images to use, if multiple LR images available per HR image.')
@click.option('--image_shortlist', default=None, help='Location of text file containing image names'
                                                      ' to select from target folder')
@click.option('--data_split', default=None,
              help='Specifies data split to extract (train/test/eval).  Defaults to eval if not specified.')
@click.option('--metadata_file', default=None, help='Location of datafile containing metadata information.'
                                                    'Defaults to degradation_metadata.csv if not specified.')
@click.option('--ignore_degradation_location', is_flag=True, help='Set to true to ignore degradation '
                                                                  'positioning information (e.g. 1-x)')
@click.option('--augmentation_normalization', default=None, multiple=True,
              help='Set to true to normalize all incoming metadata, '
                   'or specify a list of the specific attributes that should be normalized.')
@click.option("--id_source", default=None,
              help='The file to gather image IDs from.')
@click.option('--recursive', default=False,
              help='Specify whether to search for further images in sub-folders of the main lr directory.')
# Model Config
@click.option("-me", "--model_and_epoch", multiple=True,
              help='Experiments to evaluate.', type=(str, str))
@click.option("--gpu/--no-gpu", default=False,
              help='Specify whether or not to use a gpu for speeding up computations.')
@click.option("--sp_gpu", default=0,
              help='Specify specific GPU to use for computation.', show_default=True)
@click.option('--scale', default=4, help='Scale of SR to perform.', show_default=True)
# Processing/Output Config
@click.option("--results_name", default='delete_me', help='Unique folder name for this output evaluation run.')
@click.option("-m", "--metrics", multiple=True, default=None,
              help='The metrics to calculate on provided test set.')
@click.option('--save_im', is_flag=True, help='Set this flag to save all generated SR images to results folder.')
@click.option("--face_rec_profiling", is_flag=True, help='Set this flag to evaluate FR stats on given images.')
@click.option('--model_only', is_flag=True, help='Set this flag to skip all metrics and simply output results.')
@click.option('--model_loc', default=results_directory, help='Model save location for loading.')
@click.option("--out_loc", default=results_directory, help='Output directory')
@click.option('--no_image_comparison', is_flag=True,
              help='Set this flag to prevent any image comparisons being generated.')
@click.option('--save_raw_features', is_flag=True,
              help='Set this flag to save raw features generated by face recognition network.')
@click.option('--num_image_save', default=100000,
              help='Set the maximum number of images to save when running comparisons.', show_default=True)
@click.option('--save_data_model_folders', is_flag=True,
              help='Set this flag to have FR metrics saved directly to the original model folders.')
@click.option('--time_models/--no-time_models', default=True,
              help='Specify whether time model execution.  Defaults to on.')
@click.option('--data_type', default='single-frame',
              help="Determine if using SISR ('single-frame') or VSR/multi-frame ('multi-frame')", show_default=True)
@click.option('--num_frames', default=3,
              help="Set the number of frames to be grouped when 'data_type' is set to 'multi-frame'.",
              show_default=True)
@click.option('--hr_selection', default=1,
              help="Set the frame number to use as HR (GT) when 'data_type' is set to 'multi-frame'.",
              show_default=True)
@click.option('--in_features', default=3,
              help="Set the number of features (channels) of the input image (==3 for RGB images)", show_default=True)
@click.option('--run_lpips_on_gpu', is_flag=True,
              help='Set this flag to run LPIPS metrics on GPU.')
@click.option('--lanczos_upsample', is_flag=True,
              help='Set this flag to generate an additional LR image based on Lanczos upsampling.')
@click.option('--use_mps', is_flag=True,
              help='Set this flag to use MPS as GPU device.')
@click_config_file.configuration_option(provider=toml_provider, implicit=False)
def eval_run(**kwargs):
    """
    Main function that controls the creation, configuration and running of a SISR evaluation experiment.
    All functionality can be controlled via the CONFIG toml file (descriptions of available parameters provided in evaluation/standard_eval.py)
    """
    from rumpy.sr_tools.helper_functions import create_dir_if_empty
    from rumpy.shared_framework.evaluation.standard_eval import EvalHub

    out_dir = os.path.join(kwargs['out_loc'], kwargs['results_name'])
    create_dir_if_empty(out_dir)

    if type(kwargs['group_select']) == tuple:
        kwargs['group_select'] = list(kwargs['group_select'])

    if type(kwargs['augmentation_normalization']) == tuple:
        kwargs['augmentation_normalization'] = list(kwargs['augmentation_normalization'])

    if kwargs['use_mps']:
        kwargs['sp_gpu'] = 'mps'
    del kwargs['use_mps']

    input_params = {**kwargs}
    if kwargs['model_only']:
        input_params['metrics'] = None

    with open(os.path.join(out_dir, 'config.toml'), 'w') as f:
        toml.dump(input_params, f)

    eval_hub = EvalHub(**kwargs)

    eval_hub.full_image_protocol()


if __name__ == '__main__':
    eval_run(sys.argv[1:])
