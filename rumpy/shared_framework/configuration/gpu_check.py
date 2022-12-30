import torch
import os


def mps_check(device):
    try:
        if device == torch.device('mps') or device == 'mps':
            return True
        else:
            return False
    except:  # older pytorch versions won't have torch.device('mps') available
        return False


def device_selector(gpu, sp_device):
    if gpu != 'off':
        if torch.cuda.is_available():
            device = sp_device
        elif sp_device == 'mps' and torch.backends.mps.is_available():  # for Apple GPUs
            device = sp_device
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    return device
