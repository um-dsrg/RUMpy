import torch
import pytest
from rumpy.SISR.models.interface import SISRInterface
import sys


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


def test_srcnn_gpu(dummy_image, tmp_path):  # tmp_path is a temporary location which is deleted after running
    internal_params = {'name': 'srcnn', 'internal_params': {}}

    try:
        model = sisr_model_setup(internal_params, directory=tmp_path, gpu=True)
        result, _, _, _ = model.net_run_and_process(dummy_image)
    except:
        print('Model did not run properly with provided GPU.  This is usually due to issues with MPS (if using a Mac).')
        assert False

    assert result.shape == (1, 3, 16, 16)


def test_rcan_gpu(dummy_image, tmp_path):
    internal_params = {'name': 'rcan', 'internal_params': {}}

    try:
        model = sisr_model_setup(internal_params, directory=tmp_path, gpu=True)
        result, _, _, _ = model.net_run_and_process(dummy_image)
    except:
        print('Model did not run properly with provided GPU.  This is usually due to issues with MPS (if using a Mac).')
        assert False

    assert result.shape == (1, 3, 64, 64)


def test_rcan_dan_gpu(dummy_image, tmp_path):
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

    try:
        model = sisr_model_setup(internal_params, directory=tmp_path, gpu=True)
        result, _, _, _ = model.net_run_and_process(dummy_image)
    except:
        print('Model did not run properly with provided GPU.  This is usually due to issues with MPS (if using a Mac).')
        assert False

    assert result.shape == (1, 3, 64, 64)


def test_rcan_contrastive_gpu(dummy_image, tmp_path):
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

    try:
        model = sisr_model_setup(internal_params, directory=tmp_path, gpu=True)
        result, _, _, _ = model.net_run_and_process(dummy_image)
    except:
        print('Model did not run properly with provided GPU.  This is usually due to issues with MPS (if using a Mac).')
        assert False

    assert result.shape == (1, 3, 64, 64)


@pytest.mark.skipif(sys.platform == 'darwin', reason="Does not work on Mac MPS.")
def test_han_gpu(dummy_image, tmp_path):
    internal_params = {'name': 'han', 'internal_params': {}}

    try:
        model = sisr_model_setup(internal_params, directory=tmp_path, gpu=True)
        result, _, _, _ = model.net_run_and_process(dummy_image)
    except:
        print('Model did not run properly with provided GPU.  This is usually due to issues with MPS (if using a Mac).')
        assert False

    assert result.shape == (1, 3, 64, 64)


@pytest.mark.skipif(sys.platform == 'darwin', reason="Does not work on Mac MPS.")
def test_elan_gpu(dummy_image, tmp_path):
    internal_params = {'name': 'elan', 'internal_params': {}}

    try:
        model = sisr_model_setup(internal_params, directory=tmp_path, gpu=True)
        result, _, _, _ = model.net_run_and_process(dummy_image)
    except:
        print('Model did not run properly with provided GPU.  This is usually due to issues with MPS (if using a Mac).')
        assert False

    assert result.shape == (1, 3, 64, 64)


@pytest.mark.skipif(sys.platform == 'darwin', reason="Does not work on Mac MPS.")
def test_realesrgan_gpu(dummy_image, tmp_path):
    internal_params = {'name': 'realesrgan', 'internal_params': {}}

    try:
        model = sisr_model_setup(internal_params, directory=tmp_path, gpu=True)
        result, _, _, _ = model.net_run_and_process(dummy_image)
    except:
        print('Model did not run properly with provided GPU.  This is usually due to issues with MPS (if using a Mac).')
        assert False

    assert result.shape == (1, 3, 64, 64)
