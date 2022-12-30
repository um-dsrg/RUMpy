import torch
import torch.nn as nn


# all code originated from https://github.com/yuanjunchai/IKC/
class Predictor(nn.Module):
    def __init__(self, in_nc=3, nf=64, code_length=10, use_bias=True, **kwargs):
        super(Predictor, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(in_nc, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, code_length, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])
        #   self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.globalPooling = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, input):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        return flat.view(flat.size()[:2])  # torch size: [B, code_len]


class Corrector(nn.Module):
    def __init__(self, in_nc=3, nf=64, code_length=10, use_bias=True, **kwargs):
        super(Corrector, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(in_nc, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.code_dense = nn.Sequential(*[
            nn.Linear(code_length, nf, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Linear(nf, nf, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.global_dense = nn.Sequential(*[
            nn.Conv2d(nf * 2, nf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, code_length, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ])

        self.nf = nf
        self.globalPooling = nn.AdaptiveAvgPool2d([1, 1])

    def forward(self, input, code, res=False):
        conv_input = self.ConvNet(input)
        B, C_f, H_f, W_f = conv_input.size() # LR_size

        conv_code = self.code_dense(code).view((B, self.nf, 1, 1)).expand((B, self.nf, H_f, W_f)) # h_stretch
        conv_mid = torch.cat((conv_input, conv_code), dim=1)
        code_res = self.global_dense(conv_mid)

        # Delta_h_p
        flat = self.globalPooling(code_res)
        Delta_h_p = flat.view(flat.size()[:2])

        if res:
            return Delta_h_p
        else:
            return Delta_h_p + code

