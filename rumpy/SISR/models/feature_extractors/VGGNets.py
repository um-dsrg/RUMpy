import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vgg19
import rumpy.shared_framework.configuration.constants as sconst


# modified from http://www.robots.ox.ac.uk/~albanie/pytorch-models.html
class VggFace(nn.Module):

    def __init__(self, weights=sconst.vggface_weights, mode='recognition'):
        super(VggFace, self).__init__()
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.preprocessor = transforms.Normalize(mean=self.meta['mean'], std=self.meta['std'])
        self.mode = mode
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)

        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)
        self.final_soft = nn.Softmax()
        state_dict = torch.load(weights)
        self.load_state_dict(state_dict)
        self.eval()

    def preprocess(self, batch):
        return self.preprocessor(batch)

    def forward(self, x0):
        with torch.no_grad():
            x1 = self.conv1_1(x0)
            x2 = self.relu1_1(x1)
            x3 = self.conv1_2(x2)
            x4 = self.relu1_2(x3)
            x5 = self.pool1(x4)
            x6 = self.conv2_1(x5)
            x7 = self.relu2_1(x6)
            x8 = self.conv2_2(x7)
            x9 = self.relu2_2(x8)
            x10 = self.pool2(x9)
            x11 = self.conv3_1(x10)
            x12 = self.relu3_1(x11)
            x13 = self.conv3_2(x12)
            x14 = self.relu3_2(x13)
            if self.mode == 'ReLU32' or self.mode == 'ReLU3_2' or self.mode == 'relu32' or self.mode == 'relu3_2':
                return x14
            x15 = self.conv3_3(x14)
            x16 = self.relu3_3(x15)
            x17 = self.pool3(x16)
            x18 = self.conv4_1(x17)
            x19 = self.relu4_1(x18)
            x20 = self.conv4_2(x19)
            x21 = self.relu4_2(x20)
            x22 = self.conv4_3(x21)
            x23 = self.relu4_3(x22)
            x24 = self.pool4(x23)
            x25 = self.conv5_1(x24)
            x26 = self.relu5_1(x25)
            x27 = self.conv5_2(x26)
            x28 = self.relu5_2(x27)
            x29 = self.conv5_3(x28)
            if self.mode == 'p_loss':  # final conv layer pre final pooling layer
                return x29
            x30 = self.relu5_3(x29)
            x31_preflatten = self.pool5(x30)
            x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
            x32 = self.fc6(x31)
            x33 = self.relu6(x32)
            x34 = self.dropout6(x33)
            x35 = self.fc7(x34)
            x36 = self.relu7(x35)
            if self.mode == 'recognition':  # final FC layer prior to class layer
                return x36
            x37 = self.dropout7(x36)
            x38 = self.fc8(x37)
            return self.final_soft(x38)


# modified from https://github.com/eriklindernoren/PyTorch-GAN
class VGGFeatureExtractor(nn.Module):
    def __init__(self, normalise_input=True, device=torch.device('cpu'), mode='p_loss'):
        super(VGGFeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device=device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device=device)
        self.normalise_input = normalise_input

        self.list_outputs = isinstance(mode, list)

        if self.list_outputs:
            # Adapted from here: https://github.com/cszn/KAIR/blob/06bd194f4cd0da6f4159c058b6fb5b9be689c18a/models/loss.py#L54
            self.vgg19_out = nn.Sequential()

            mode = [-1] + mode
            for i in range(len(mode)-1):
                self.vgg19_out.add_module('child'+str(i), nn.Sequential(*list(vgg19_model.features.children())[(mode[i]+1):(mode[i+1]+1)]))
        else:
            vgg19_layer_names_full =  [
                'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
                'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
                'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
                'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
            ]

            vgg19_layer_names_no_underscore = [l.replace('_', '') for l in vgg19_layer_names_full]

            # Default behaviour is p_loss so just set the output to it
            # VGG19 conv5_4 or 54
            self.vgg19_out = nn.Sequential(*list(vgg19_model.features)[:35])

            layer_index = -1

            if mode in vgg19_layer_names_full:
                for index, vgg_layer in enumerate(vgg19_layer_names_full):
                    if mode == vgg_layer:
                        layer_index = index

            if mode in vgg19_layer_names_no_underscore:
                for index, vgg_layer in enumerate(vgg19_layer_names_no_underscore):
                    if mode == vgg_layer:
                        layer_index = index

            if layer_index != -1:
                self.vgg19_out = nn.Sequential(*list(vgg19_model.features)[:layer_index+1])

        for k, v in self.vgg19_out.named_parameters():
            v.requires_grad = False

    def forward(self, img):
        if self.normalise_input:
            img = (img - self.mean)/self.std

        if self.list_outputs:
            output = []
            for child_model in self.vgg19_out.children():
                img = child_model(img)
                output.append(img.clone())
            return output
        else:
            return self.vgg19_out(img)
