import click
import sys
from rumpy.image_tools import available_tools


@click.command()
@click.option("--source_dir",
              help='Input directory to source images.')
@click.option("--output_dir",
              help='Output directory to save new images.')
@click.option("--seed", default=8, help='Random seed.')
@click.option("--scale", default=4.0, help='Downsampling/upsampling scale factor.')
@click.option("--pipeline_config", default=None,
              help='location of TOML image pipeline configuration, if available.')
@click.option('--recursive', is_flag=True,
              help='Set this flag to signal data converter to seek out all images in all sub-directories of '
                   'directory specified.')
@click.option("--pipeline", help='Pipeline of operations to perform, separated by "-". '
                                 'Available operations: %s' % ', '.join(list(available_tools.keys())),
              show_default=True)
@click.option('--output_extension', default='.png',
              help='Define image extension to be used for all output images.')
@click.option('--multiples', default=1,
              help='Number of copies to generate from each image.')
@click.option('--override_cli', default=False,
              help='Override default values and command-line inputs with pipeline config.')
def image_manipulator(**kwargs):
    """
    CLI function for applying transformations to images.  Place requested operations in the --pipeline parameter
    or alternatively specify more complicated configurations in a .toml file and read in via --pipeline_config.

    TODO: add example.
    """
    from rumpy.image_tools.image_pipeline import pipeline_prep_and_run
    pipeline_prep_and_run(**kwargs)


if __name__ == '__main__':
    image_manipulator(sys.argv[1:])  # for use when debugging with pycharm
