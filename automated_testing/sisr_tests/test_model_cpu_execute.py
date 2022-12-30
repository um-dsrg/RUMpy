import torch
import pytest
from rumpy.SISR.models.interface import SISRInterface


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


def sisr_model_setup(params, directory, gpu=False):
    if gpu:
        gpu_type = 'single'
    else:
        gpu_type = 'off'

    model = SISRInterface(model_loc=directory, experiment='test',
                          gpu=gpu_type, sp_gpu=model_setup_device,
                          mode='train', new_params=params,
                          no_directories=True)
    return model


def test_srcnn(dummy_image, tmp_path):  # tmp_path is a temporary location which is deleted after running
    internal_params = {'name': 'srcnn', 'internal_params': {}}

    model = sisr_model_setup(internal_params, directory=tmp_path)
    result, _, _, _ = model.net_run_and_process(dummy_image)

    assert result.shape == (1, 3, 16, 16)


def test_rcan(dummy_image, tmp_path):
    internal_params = {'name': 'rcan', 'internal_params': {}}

    model = sisr_model_setup(internal_params, directory=tmp_path)

    result, _, _, _ = model.net_run_and_process(dummy_image)

    assert result.shape == (1, 3, 64, 64)


def test_rcan_dan(dummy_image, tmp_path):
    internal_params = {'name': 'dan', 'internal_params': {'mode': "v1QRCAN",
                                                          'scale': 4,
                                                          'metadata': ["blur_kernel", ],
                                                          'selective_meta_blocks': [True, False, False, False, False,
                                                                                    False, False, False, False,
                                                                                    False, ],
                                                          'style': "standard",
                                                          'include_q_layer': True,
                                                          'num_q_layers_inner_residual': 1,
                                                          'ignore_degradation_location': True}}

    model = sisr_model_setup(internal_params, directory=tmp_path)

    result, _, _, _ = model.net_run_and_process(dummy_image)

    assert result.shape == (1, 3, 64, 64)


def test_rcan_contrastive(dummy_image, tmp_path):
    internal_params = {'name': 'ContrastiveBlindQRCAN', 'internal_params': {'scale': 4,
                                                                            'selective_meta_blocks': [True, False,
                                                                                                      False, False,
                                                                                                      False,
                                                                                                      False, False,
                                                                                                      False, False,
                                                                                                      False, ],
                                                                            'style': "standard",
                                                                            'block_encoder_loading': True,
                                                                            'include_q_layer': True,
                                                                            'num_q_layers_inner_residual': 1, }}

    model = sisr_model_setup(internal_params, directory=tmp_path)

    result, _, _, _ = model.net_run_and_process(dummy_image)

    assert result.shape == (1, 3, 64, 64)


def test_han(dummy_image, tmp_path):
    internal_params = {'name': 'han', 'internal_params': {}}

    model = sisr_model_setup(internal_params, directory=tmp_path)

    result, _, _, _ = model.net_run_and_process(dummy_image)

    assert result.shape == (1, 3, 64, 64)


def test_elan(dummy_image, tmp_path):
    internal_params = {'name': 'elan', 'internal_params': {}}

    model = sisr_model_setup(internal_params, directory=tmp_path)

    result, _, _, _ = model.net_run_and_process(dummy_image)

    assert result.shape == (1, 3, 64, 64)


def test_realesrgan(dummy_image, tmp_path):
    internal_params = {'name': 'realesrgan', 'internal_params': {}}

    model = sisr_model_setup(internal_params, directory=tmp_path)

    result, _, _, _ = model.net_run_and_process(dummy_image)

    assert result.shape == (1, 3, 64, 64)
