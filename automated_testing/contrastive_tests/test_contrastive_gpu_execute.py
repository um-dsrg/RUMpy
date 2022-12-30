import torch
import pytest
from rumpy.regression.models.interface import RegressionInterface


@pytest.fixture
def dummy_image():
    dummy_im = torch.rand((1, 3, 16, 16), dtype=torch.float32)
    return dummy_im


if torch.cuda.is_available():
    model_setup_device = 0
elif torch.backends.mps.is_available():
    model_setup_device = 'mps'
else:
    model_setup_device = 'cpu'


def contrastive_model_setup(params, directory, gpu=False):
    if gpu:
        gpu_type = 'single'
    else:
        gpu_type = 'off'

    model = RegressionInterface(model_loc=directory, experiment='test',
                                gpu=gpu_type, sp_gpu=model_setup_device,
                                mode='train', new_params=params,
                                no_directories=True)
    return model


def test_supmoco_gpu(dummy_image, tmp_path):  # tmp_path is a temporary location which is deleted after running
    internal_params = {'name': 'supmoco', 'internal_params': {'crop_count': 4,
                                                              'model_name': "default"}}

    model = contrastive_model_setup(internal_params, directory=tmp_path, gpu=True)
    result, _, _ = model.net_run_and_process(dummy_image)

    assert result[0].shape == (1, 256)


def test_moco_gpu(dummy_image, tmp_path):  # tmp_path is a temporary location which is deleted after running
    internal_params = {'name': 'mococontrastive', 'internal_params': {'crop_count': 4,
                                                                      'model_name': "default"}}

    model = contrastive_model_setup(internal_params, directory=tmp_path, gpu=True)
    result, _, _ = model.net_run_and_process(dummy_image)

    assert result[0].shape == (1, 256)


def test_weakcon_gpu(dummy_image, tmp_path):  # tmp_path is a temporary location which is deleted after running
    internal_params = {'name': 'weakcon', 'internal_params': {'crop_count': 4,
                                                              'model_name': "default"}}

    model = contrastive_model_setup(internal_params, directory=tmp_path, gpu=True)
    result, _, _ = model.net_run_and_process(dummy_image)

    assert result[0].shape == (1, 256)
