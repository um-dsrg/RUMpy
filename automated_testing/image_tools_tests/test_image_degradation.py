import pytest
import os
from click.testing import CliRunner

from rumpy.image_tools.click_cli import image_manipulator

runner = CliRunner()


@pytest.fixture
def full_pipeline_test_config():
    return os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir)), 'blur_downsample_noise_compress.toml')


def test_full_routine(full_pipeline_test_config):
    input_dir = os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir)), 'hr_examples')
    output_dir = os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir)), 'lr_test_outputs')

    response = runner.invoke(image_manipulator,
                             ['--pipeline_config', full_pipeline_test_config, '--source_dir', input_dir, '--output_dir',
                              output_dir])
    print('\n')
    print(response.output)
    assert response.exit_code == 0
