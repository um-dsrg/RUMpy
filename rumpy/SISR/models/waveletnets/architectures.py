import torch
from torch import nn
import math
import pickle
import os


def identity_loss(feat_orig, feat_pred):
    loss = []
    fn = nn.L1Loss()
    for orig, pred in zip(feat_orig, feat_pred):
        loss.append(fn(orig, pred)/(orig.numel()/orig.size()[0]))

    return sum(loss)


def loss_MSE(x, y, size_average=False):
    z = x - y
    z2 = z * z
    if size_average:
        return z2.mean()
    else:
        return z2.sum().div(x.size(0) * 2)


# WaveletSRNet code adapted from https://github.com/RemyEE/WaveletSRNet
def loss_Textures(x, y, nc=3, alpha=1.2, margin=0):
    xi = x.contiguous().view(x.size(0), -1, nc, x.size(2), x.size(3))
    yi = y.contiguous().view(y.size(0), -1, nc, y.size(2), y.size(3))

    xi2 = torch.sum(xi * xi, dim=2)
    yi2 = torch.sum(yi * yi, dim=2)

    out = nn.functional.relu(yi2.mul(alpha) - xi2 + margin)

    return torch.mean(out)


class WaveletDiscriminator(nn.Module):

    def __init__(self, scale):
        super(WaveletDiscriminator, self).__init__()
        interim_base_c = 32
        self.end_base_c = 256
        operator = int(math.log2(scale))
        wavelet_channels = int(math.pow(4, operator))

        self.embedding = self.construct_embedding(inc=wavelet_channels*3, interimc=interim_base_c*wavelet_channels,
                                                  outc=self.end_base_c*wavelet_channels, groups=wavelet_channels)
        self.prediction = nn.Conv2d(in_channels=self.end_base_c, out_channels=1, kernel_size=3, stride=1, padding=1,
                                    bias=True)

    def construct_embedding(self, inc, interimc, outc, groups):
        layers = [
            nn.Conv2d(in_channels=inc, out_channels=interimc, kernel_size=3, stride=2,
                      padding=1, groups=groups, bias=True),
            nn.BatchNorm2d(interimc),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=interimc, out_channels=outc, kernel_size=3, stride=1,
                      padding=1, groups=groups, bias=True),
            nn.BatchNorm2d(outc),
            nn.LeakyReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        embedding = self.embedding(x)
        shape = embedding.size()
        concat = embedding.reshape(shape[0], -1, self.end_base_c, shape[2], shape[3]).sum(dim=1)
        return self.prediction(concat)


class WaveletTransform(nn.Module):
    def __init__(self, scale=1, dec=True, params_path=os.path.join(os.path.dirname(__file__), 'wavelet_weights.pkl'), transpose=True):
        super(WaveletTransform, self).__init__()

        self.scale = int(math.log2(scale))
        self.dec = dec
        self.transpose = transpose

        ks = int(math.pow(2, self.scale))
        nc = 3 * ks * ks

        if dec:
            self.conv = nn.Conv2d(in_channels=3, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=3,
                                  bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0,
                                           groups=3, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = open(params_path, 'rb')
                dct = pickle.load(f, encoding='latin1')
                f.close()
                m.weight.data = torch.from_numpy(dct['rec%d' % ks])
                m.weight.requires_grad = False

    def forward(self, x):
        if self.dec:
            output = self.conv(x)
            if self.transpose:
                osz = output.size()
                # print(osz)
                output = output.view(osz[0], 3, -1, osz[2], osz[3]).transpose(1, 2).contiguous().view(osz)
        else:
            if self.transpose:
                xx = x
                xsz = xx.size()
                xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]).transpose(1, 2).contiguous().view(xsz)
            output = self.conv(xx)
        return output


class _Residual_Block(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1):
        super(_Residual_Block, self).__init__()

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output, identity_data)))
        return output


def make_layer(block, num_of_layer, inc=64, outc=64, groups=1):
    layers = []
    layers.append(block(inc=inc, outc=outc, groups=groups))
    for _ in range(1, num_of_layer):
        layers.append(block(inc=outc, outc=outc, groups=groups))
    return nn.Sequential(*layers)


class _Interim_Block(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1):
        super(_Interim_Block, self).__init__()

        self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1,
                                     bias=False)
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity_data = self.conv_expand(x)
        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output, identity_data)))
        return output


class WaveletSRNet(nn.Module):
    def __init__(self, scale=2, num_layers_res=2):
        super(WaveletSRNet, self).__init__()

        self.scale = int(math.log2(scale))
        self.groups = int(math.pow(4, self.scale))
        self.wavelet_c = wavelet_c = 32

        # ----------input conv-------------------
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(64)
        self.relu_input = nn.ReLU(inplace=True)

        # ----------residual-------------------
        self.residual = nn.Sequential(
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=64),
            make_layer(_Residual_Block, num_layers_res, inc=64, outc=128),
            make_layer(_Residual_Block, num_layers_res, inc=128, outc=256),
            make_layer(_Residual_Block, num_layers_res, inc=256, outc=512),
            make_layer(_Residual_Block, num_layers_res, inc=512, outc=1024)
        )

        # ----------wavelet conv-------------------
        inc = 1024
        layer_num = 1
        if self.scale >= 0:
            g = 1
            self.interim_0 = _Interim_Block(inc, wavelet_c * g, g)
            self.wavelet_0 = make_layer(_Residual_Block, layer_num, wavelet_c * g, wavelet_c * 2 * g, g)
            self.predict_0 = nn.Conv2d(in_channels=wavelet_c * 2 * g, out_channels=3 * g, kernel_size=3, stride=1,
                                       padding=1,
                                       groups=g, bias=True)

        if self.scale >= 1:
            g = 3
            self.interim_1 = _Interim_Block(inc, wavelet_c * g, g)
            self.wavelet_1 = make_layer(_Residual_Block, layer_num, wavelet_c * g, wavelet_c * 2 * g, g)
            self.predict_1 = nn.Conv2d(in_channels=wavelet_c * 2 * g, out_channels=3 * g, kernel_size=3, stride=1,
                                       padding=1,
                                       groups=g, bias=True)

        if self.scale >= 2:
            g = 12
            self.interim_2 = _Interim_Block(inc, wavelet_c * g, g)
            self.wavelet_2 = make_layer(_Residual_Block, layer_num, wavelet_c * g, wavelet_c * 2 * g, g)
            self.predict_2 = nn.Conv2d(in_channels=wavelet_c * 2 * g, out_channels=3 * g, kernel_size=3, stride=1,
                                       padding=1,
                                       groups=g, bias=True)

        if self.scale >= 3:
            g = 48
            self.interim_3 = _Interim_Block(inc, wavelet_c * g, g)
            self.wavelet_3 = make_layer(_Residual_Block, layer_num, wavelet_c * g, wavelet_c * 2 * g, g)
            self.predict_3 = nn.Conv2d(in_channels=wavelet_c * 2 * g, out_channels=3 * g, kernel_size=3, stride=1,
                                       padding=1,
                                       groups=g, bias=True)

        if self.scale >= 4:
            g = 192
            self.interim_4 = _Interim_Block(inc, wavelet_c * g, g)
            self.wavelet_4 = make_layer(_Residual_Block, layer_num, wavelet_c * g, wavelet_c * 2 * g, g)
            self.predict_4 = nn.Conv2d(in_channels=wavelet_c * 2 * g, out_channels=3 * g, kernel_size=3, stride=1,
                                       padding=1,
                                       groups=g, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

        # model converting wavelet approximation to image, params are frozen.
        self.wavelet_rec = WaveletTransform(scale=scale, dec=False)
        for param in self.wavelet_rec.parameters():  # Redundant, but just in case...
            param.requires_grad = False

    def wavelet_predict(self, x):

        f = self.relu_input(self.bn_input(self.conv_input(x)))

        f = self.residual(f)

        if self.scale >= 0:
            out_0 = self.interim_0(f)
            out_0 = self.wavelet_0(out_0)
            out_0 = self.predict_0(out_0)
            out = out_0

        if self.scale >= 1:
            out_1 = self.interim_1(f)
            out_1 = self.wavelet_1(out_1)
            out_1 = self.predict_1(out_1)
            out = torch.cat((out, out_1), 1)

        if self.scale >= 2:
            out_2 = self.interim_2(f)
            out_2 = self.wavelet_2(out_2)
            out_2 = self.predict_2(out_2)
            out = torch.cat((out, out_2), 1)

        if self.scale >= 3:
            out_3 = self.interim_3(f)
            out_3 = self.wavelet_3(out_3)
            out_3 = self.predict_3(out_3)
            out = torch.cat((out, out_3), 1)

        if self.scale >= 4:
            out_4 = self.interim_4(f)
            out_4 = self.wavelet_4(out_4)
            out_4 = self.predict_4(out_4)
            out = torch.cat((out, out_4), 1)

        return out

    def forward(self, x, train=False):
        out = self.wavelet_predict(x)
        if train:
            return out, self.wavelet_rec(out)
        else:
            return self.wavelet_rec(out)

    def reset_parameters(self):
        pass  # TODO: implement if required
