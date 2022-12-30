# adapted from https://github.com/AlfredXiangWu/LightCNN/blob/master/light_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import rumpy.shared_framework.configuration.constants as sconst


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def full_extraction(self, x):
        return self.conv_a(x), self.conv(self.conv_a(x))

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class network_29layers(nn.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers, self).__init__()
        self.conv1  = mfm(1, 48, 5, 1, 2)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc     = mfm(8*8*128, 256, type=0)
        self.fc2    = nn.Linear(256, num_classes)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        fc = F.dropout(fc, training=self.training)
        out = self.fc2(fc)
        return out, fc

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        c1, c2 = self.group4.full_extraction(x)
        return [c1, c2]

    def preprocess(self, x):
        size_check = F.interpolate(x, size=(128, 128), mode='bicubic')
        grayscale = 0.299 * size_check[:, 0, :, :] + 0.587 * size_check[:, 1, :, :] + 0.114 * size_check[:, 2, :, :]
        return grayscale.unsqueeze(1)


def LightCNN_29Layers(device=torch.device('cpu')):  # TODO: Remove dependency on nn.dataparallel
    model = network_29layers(resblock, [1, 2, 3, 4])
    model.to(device)
    model = nn.DataParallel(model)
    weights = torch.load(sconst.lightcnn_weights, map_location=device)
    model.load_state_dict(weights['state_dict'])
    model.eval()
    return model
