import pytest
import os
from click.testing import CliRunner
import toml
import shutil

from rumpy.shared_framework.net_train import experiment_setup

runner = CliRunner()

file_loc = os.path.abspath(os.path.join(__file__, os.path.pardir))


@pytest.fixture
def rcan_div2k_config():
    return os.path.join(file_loc, 'rcan_testing_config.toml')


@pytest.mark.slow
def test_basic_training(rcan_div2k_config):
    params = toml.load(rcan_div2k_config)
    params['experiment_save_loc'] = file_loc

    params['data']['training_sets']['data_1']['lr'] = os.path.join(file_loc, 'training_dataset',
                                                                   'lr_div2k_reduced_blur_noise_compress')
    params['data']['training_sets']['data_1']['hr'] = os.path.join(file_loc, 'training_dataset',
                                                                   'HR_div2k_reduced')

    params['data']['eval_sets']['data_1']['lr'] = os.path.join(file_loc, 'eval_dataset', 'lr')
    params['data']['eval_sets']['data_1']['hr'] = os.path.join(file_loc, 'eval_dataset', 'hr')

    with open(rcan_div2k_config, 'w') as f:
        toml.dump(params, f)

    response = runner.invoke(experiment_setup, ['--parameters', rcan_div2k_config, '--num_epochs', '1'])
    shutil.rmtree(os.path.join(file_loc, 'test_rcan'))
    print('\n')
    print(response.output)
    assert response.exit_code == 0
