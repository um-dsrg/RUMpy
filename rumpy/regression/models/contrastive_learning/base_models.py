import torch
from torch import nn as nn
from torch.nn import functional as F


class AdjustedStandardModel(torch.nn.Module):
    def __init__(self, chosen_model, dropdown_q=None):
        super(AdjustedStandardModel, self).__init__()

        self.chosen_model_class_name = chosen_model.__class__.__name__.lower()

        # Get all the layers before the FC layer
        self.features = nn.Sequential(*list(chosen_model.children())[:-1])

        dropdown_input_dim = 0

        # Get the size of the input dimension for the FC/classifier layer
        # FC attribute is found in ResNet models
        if hasattr(chosen_model, 'fc'):
            self.has_classifier = False
            mlp_input_dim = chosen_model.fc.in_features
            dropdown_input_dim = chosen_model.fc.out_features

            # Add the extra FC layer like for MoCoV2
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_input_dim),
                nn.ReLU(),
                chosen_model.fc
            )

        # Classifier attributes is found in most of the other models, this will mainly be for DenseNet
        elif hasattr(chosen_model, 'classifier'):
            self.has_classifier = True

            # Add the extra FC layer like for MoCoV2
            if 'densenet' in self.chosen_model_class_name:
                mlp_input_dim = chosen_model.classifier.in_features
                dropdown_input_dim = chosen_model.classifier.out_features

                self.mlp = nn.Sequential(
                    nn.Linear(mlp_input_dim, mlp_input_dim),
                    nn.ReLU(),
                    chosen_model.classifier
                )

            elif 'efficientnet' in self.chosen_model_class_name:
                classifier_input_dim = chosen_model.classifier[1].in_features  # Sequential(Dropout, Linear), Linear is the 2nd layer
                classifier_output_dim = chosen_model.classifier[1].out_features
                dropdown_input_dim = classifier_output_dim

                # Original EfficientNet Code for the classifier:
                # https://github.com/pytorch/vision/blob/59ef2ab0ef357d9c4d7d6e72c363901d3cf05382/torchvision/models/efficientnet.py#L326-L329
                self.mlp = nn.Sequential(
                    chosen_model.classifier[0],
                    nn.Linear(classifier_input_dim, classifier_input_dim), # In the original: nn.Linear(lastconv_output_channels, num_classes)
                    nn.ReLU(),
                    nn.Linear(classifier_input_dim, classifier_output_dim)
                )
            elif 'mobilenetv3' in self.chosen_model_class_name:
                classifier_last_input_dim = chosen_model.classifier[3].in_features  # Sequential(Linear, Hardswish, Dropout, Linear), Linear is the 4th layer
                classifier_last_output_dim = chosen_model.classifier[3].out_features
                dropdown_input_dim = classifier_last_output_dim

                # Original MobileNetV3 Code for the classifier:
                # https://github.com/pytorch/vision/blob/59ef2ab0ef357d9c4d7d6e72c363901d3cf05382/torchvision/models/efficientnet.py#L326-L329
                self.mlp = nn.Sequential(
                    chosen_model.classifier[0],
                    chosen_model.classifier[1],
                    chosen_model.classifier[2],
                    nn.Linear(classifier_last_input_dim, classifier_last_input_dim), # In the original: nn.Linear(last_channel, num_classes)
                    nn.ReLU(),
                    nn.Linear(classifier_last_input_dim, classifier_last_output_dim)
                )

        if dropdown_q is not None:
            self.drop_mlp = nn.Sequential(
                nn.Linear(dropdown_input_dim, 64),
                nn.LeakyReLU(0.1, True),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.1, True),
                nn.Linear(32, dropdown_q),
            )
            self.dropdown = True
        else:
            self.dropdown = False


    def forward(self, x):
        # Split the model into 2 parts to get both the output features and the MLP output
        fea = self.features(x).squeeze(-1).squeeze(-1)

        if self.has_classifier:
            if 'densenet' in self.chosen_model_class_name:
                # https://github.com/pytorch/vision/blob/64b1e279d7963c923bd453de07589a25cb6e8d03/torchvision/models/densenet.py#L215-L217
                fea = F.relu(fea.unsqueeze(-1).unsqueeze(-1), inplace=True)
                fea = F.adaptive_avg_pool2d(fea, (1, 1))
                fea = torch.flatten(fea, 1)
            elif 'efficientnet' in self.chosen_model_class_name or 'mobilenetv3' in self.chosen_model_class_name:
                # https://github.com/pytorch/vision/blob/59ef2ab0ef357d9c4d7d6e72c363901d3cf05382/torchvision/models/efficientnet.py#L345-L350
                fea = F.adaptive_avg_pool2d(fea.unsqueeze(-1).unsqueeze(-1), (1, 1))
                fea = torch.flatten(fea, 1)

        if 'shufflenetv2' in self.chosen_model_class_name:
            # https://github.com/pytorch/vision/blob/59ef2ab0ef357d9c4d7d6e72c363901d3cf05382/torchvision/models/shufflenetv2.py#L153-L163
            fea = fea.unsqueeze(-1).unsqueeze(-1).mean([2, 3])  # globalpool
            fea = torch.flatten(fea, 1)

        out = self.mlp(fea)
        out_dict = {'q': out}

        if self.dropdown:
            out_dict['dropdown_q'] = self.drop_mlp(out)

        return fea, out_dict
