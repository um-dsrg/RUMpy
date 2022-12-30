import torch
from torch import nn


class DGFMBLayer(nn.Module):
    """
    Degradation-Guided Feature Modulation Block

    This module was defined in https://www.sciencedirect.com/science/article/pii/S0950705122004774 and uses
    an attention-based system to introduce image degradation information into an SR model.
    """

    def __init__(self, num_channels=64, degradation_full_dim=256, degradation_reduced_dim=64, num_layers=2, use_linear=False, use_reduction=True):
        super().__init__()

        self.use_linear = use_linear

        if not use_reduction:
            degradation_reduced_dim = degradation_full_dim

        combined_feat_deg_size = num_channels + degradation_reduced_dim
        attention_layers = []

        if isinstance(num_layers, list):
            in_out_layer_sizes = [combined_feat_deg_size]
            in_out_layer_sizes.extend([n for n in num_layers])
            in_out_layer_sizes.append(num_channels)

            for i in range(len(in_out_layer_sizes) - 1):
                if use_linear:
                    temp_layer = nn.Linear(num_layers[i], num_layers[i+1])
                else:
                    temp_layer = nn.Conv2d(num_layers[i], num_layers[i+1], 1, padding=0, bias=True)

                attention_layers.append(temp_layer)
        else:
            multiplier = num_layers
            layer_sizes = [combined_feat_deg_size]

            for i in range(num_layers):
                if (num_channels + degradation_reduced_dim) > 15:
                    layer_sizes.append((num_channels - combined_feat_deg_size)//multiplier + combined_feat_deg_size)
                else:
                    layer_sizes.append(num_channels//multiplier)

                if use_linear:
                    attention_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                else:
                    attention_layers.append(nn.Conv2d(layer_sizes[i], layer_sizes[i+1], 1, padding=0, bias=True))
                multiplier -= 1

        attention_layers.append(nn.Sigmoid())

        if use_linear:
            attention_layers.insert(0, nn.Flatten())

        self.attention_module = nn.Sequential(*attention_layers)

        if use_reduction:
            if use_linear:
                self.degradation_reduction = nn.Sequential(
                    nn.Linear(degradation_full_dim, degradation_reduced_dim)
                )
            else:
                self.degradation_reduction = nn.Sequential(
                    nn.Conv2d(degradation_full_dim, degradation_reduced_dim, 1, padding=0, bias=True)
                )
        else:
            self.degradation_reduction = nn.Sequential(nn.Identity())

        self.global_average_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, features, encoding):
        features_gap = self.global_average_pool(features)
        reduced_encoding = self.degradation_reduction(encoding)

        if self.use_linear:
            reduced_encoding = reduced_encoding.unsqueeze(-1).unsqueeze(-1)
            features_encoding_concat = torch.cat((features_gap, reduced_encoding), dim=1).squeeze(-1).squeeze(-1)
        else:
            features_encoding_concat = torch.cat((features_gap, reduced_encoding), dim=1)

        attention = self.attention_module(features_encoding_concat)

        if self.use_linear:
            attention = attention.unsqueeze(-1).unsqueeze(-1)

        out = (features * attention) + features

        return out
