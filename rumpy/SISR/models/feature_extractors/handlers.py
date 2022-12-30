import torch
from rumpy.SISR.models.feature_extractors import lightCNN, VGGNets


def perceptual_loss_mechanism(name, device=torch.device('cpu'), mode='recognition'):
    if name == 'vgg' and mode != 'recognition':
        mech = VGGNets.VGGFeatureExtractor(device=device, mode=mode)
    elif name == 'vggface':
        mech = VGGNets.VggFace(mode=mode)
    elif name == 'lightcnn':
        mech = lightCNN.LightCNN_29Layers(device=device)
    else:
        raise Exception('Specified feature extractor not implemented.')
    return mech
