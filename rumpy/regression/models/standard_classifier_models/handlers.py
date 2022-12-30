import torchvision.models as models
import torch.nn as nn

from rumpy.regression.models import SelectiveSoftmax, DegradationRegressor
from rumpy.regression.models.standard_classifier_models.architectures import BasicNet


class BasicNNHandler(DegradationRegressor):
    """
    Basic neural network, modified from Pytorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """
    def __init__(self, device, model_save_dir, eval_mode=False, output_size=10,
                 scheduler=None, scheduler_params=None, lr=1e-4, **kwargs):
        super(BasicNNHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                             **kwargs)
        self.net = BasicNet(output_size=output_size)
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, device=device, perceptual=None)


class ResnetHandler(DegradationRegressor):
    """
    Resnet obtained from PyTorch default model library: https://pytorch.org/vision/stable/models.html
    """

    def __init__(self, device, model_save_dir, eval_mode=False, output_size=10, input_patch_num=1,
                 model_type='resnet18', add_softmax=False, scheduler=None, scheduler_params=None, lr=1e-4, **kwargs):

        super(ResnetHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                            input_patch_num=input_patch_num, **kwargs)
        if model_type == 'resnet18':
            self.net = models.resnet18(num_classes=output_size)
        elif model_type == 'resnet50':
            self.net = models.resnet50(num_classes=output_size)
        else:
            raise RuntimeError('Model Undefined.')

        if input_patch_num > 1:  # multiple patches input into network
            self.net.conv1 = nn.Conv2d(input_patch_num*3, 64, kernel_size=7,
                                       stride=2, padding=3, bias=False)
        if add_softmax:
            self.net.fc = nn.Sequential(nn.Linear(self.net.fc.in_features, output_size),
                                        SelectiveSoftmax([0, 441]))  # TODO: this harcoding needs to be removed and a robust implementation included

        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, device=device, perceptual=None)


class EfficientnetHandler(DegradationRegressor):
    """
    Efficientnet obtained from PyTorch default model library: https://pytorch.org/vision/stable/models.html

    REQUIRES TORCHVISION 0.11
    """
    def __init__(self, device, model_save_dir, eval_mode=False, output_size=10, scheduler=None,
                 scheduler_params=None, lr=1e-4, **kwargs):
        super(EfficientnetHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                                  **kwargs)

        self.net = models.efficientnet_b3(num_classes=output_size)
        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, device=device, perceptual=None)


class DensenetHandler(DegradationRegressor):
    """
    Densenet obtained from PyTorch default model library: https://pytorch.org/vision/stable/models.html
    """
    def __init__(self, device, model_save_dir, eval_mode=False, output_size=10, add_softmax=False,
                 input_patch_num=1, scheduler=None, scheduler_params=None, lr=1e-4, **kwargs):
        super(DensenetHandler, self).__init__(device=device, model_save_dir=model_save_dir, eval_mode=eval_mode,
                                              input_patch_num=input_patch_num, **kwargs)
        self.net = models.densenet169(num_classes=output_size)
        if add_softmax:
            self.net.classifier = nn.Sequential(nn.Linear(self.net.classifier.in_features, output_size), nn.Softmax(1))

        if input_patch_num > 1:  # multiple patches input into network
            self.net.features[0] = nn.Conv2d(input_patch_num*3, 64, kernel_size=7,
                                             stride=2, padding=3, bias=False)

        self.activate_device()
        self.training_setup(lr, scheduler, scheduler_params, device=device, perceptual=None)


