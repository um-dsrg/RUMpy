# all code originally obtained from https://github.com/greatlog/DAN

import torch
import torch.nn as nn
import os


class PCAEncoder(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)
        self.size = self.weight.size()

    def forward(self, batch_kernel):
        B, H, W = batch_kernel.size()  # [B, l, l]
        return torch.bmm(
            batch_kernel.view((B, 1, H * W)), self.weight.expand((B,) + self.size)
        ).view((B, -1))


class DPCB(nn.Module):
    def __init__(self, nf1, nf2, ksize1=3, ksize2=1):
        super().__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(nf1, nf1, ksize1, 1, ksize1 // 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf1, nf1, ksize1, 1, ksize1 // 2),
        )

        self.body2 = nn.Sequential(
            nn.Conv2d(nf2, nf1, ksize2, 1, ksize2 // 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf1, nf1, ksize2, 1, ksize2 // 2),
        )

    def forward(self, x):

        f1 = self.body1(x[0])
        f2 = self.body2(x[1])

        x[0] = x[0] + torch.mul(f1, f2)
        x[1] = x[1] + f2
        return x


class DPCG(nn.Module):
    def __init__(self, nf1, nf2, ksize1, ksize2, nb):
        super().__init__()

        self.body = nn.Sequential(*[DPCB(nf1, nf2, ksize1, ksize2) for _ in range(nb)])

    def forward(self, x):
        y = self.body(x)
        y[0] = x[0] + y[0]
        y[1] = x[1] + y[1]
        return y


class Estimator(nn.Module):
    def __init__(
            self, in_nc=1, nf=64, para_len=10, num_blocks=5, scale=4, kernel_size=4, residual_form=False
    ):
        super(Estimator, self).__init__()

        self.ksize = kernel_size
        self.residual_form = residual_form
        self.head_LR = nn.Sequential(
            nn.Conv2d(in_nc, nf // 2, 5, 1, 2)
        )
        self.head_HR = nn.Sequential(
            nn.Conv2d(in_nc, nf // 2, scale * 4 + 1, scale, scale * 2),
        )

        self.body = DPCG(nf // 2, nf // 2, 3, 3, num_blocks)

        self.tail = nn.Sequential(
            nn.Conv2d(nf // 2, nf, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nf, self.ksize ** 2, 1, 1, 0),
            nn.Softmax(1),
        )

    def forward(self, GT, LR, previous_kernel=None):
        lrf = self.head_LR(LR)
        hrf = self.head_HR(GT)

        f = [lrf, hrf]
        f, _ = self.body(f)
        f = self.tail(f)
        if self.residual_form:
            return f.view(*f.size()[:2]) + previous_kernel
        else:
            return f.view(*f.size()[:2])


class Restorer(nn.Module):
    def __init__(
            self, in_nc=1, nf=64, nb=8, ng=1, scale=4, input_para=10, min=0.0, max=1.0, residual_form=False
    ):
        super(Restorer, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb
        self.residual_form = residual_form

        out_nc = in_nc

        self.head1 = nn.Conv2d(in_nc, nf, 3, stride=1, padding=1)
        self.head2 = nn.Conv2d(input_para, nf, 1, 1, 0)

        body = [DPCG(nf, nf, 3, 1, nb) for _ in range(ng)]
        self.body = nn.Sequential(*body)

        self.fusion = nn.Conv2d(nf, nf, 3, 1, 1)

        if scale == 4:  # x4
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(nf, out_nc, 3, 1, 1),
            )
        elif scale == 1:
            self.upscale = nn.Conv2d(nf, out_nc, 3, 1, 1)

        else:  # x2, x3
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale ** 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale),
                nn.Conv2d(nf, out_nc, 3, 1, 1),
            )

    def forward(self, input, ker_code, previous_sr=None):
        B, C, H, W = input.size()  # I_LR batch
        B_h, C_h = ker_code.size()  # Batch, Len=10
        ker_code_exp = ker_code.view((B_h, C_h, 1, 1))

        f1 = self.head1(input)
        f2 = self.head2(ker_code_exp)
        inputs = [f1, f2]
        f, _ = self.body(inputs)
        f = self.fusion(f)
        out = self.upscale(f)
        if self.residual_form:
            out = previous_sr + out

        return out  # torch.clamp(out, min=self.min, max=self.max)


class DANv2(nn.Module):
    def __init__(
            self,
            nf=64,
            nb=10,
            ng=5,
            in_nc=3,
            upscale=4,
            input_para=10,
            kernel_size=21,
            loop=4,
            residual_kernel=False,
            residual_sr=False,
            pca_matrix_path=os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir)), 'pca_matrix.pth'),
            **kwargs
    ):
        super(DANv2, self).__init__()

        self.ksize = kernel_size
        self.loop = loop
        self.scale = upscale
        self.residual_kernel = residual_kernel
        self.residual_sr = residual_sr

        self.Restorer = Restorer(
            nf=nf, in_nc=in_nc, nb=nb, ng=ng, scale=self.scale, input_para=input_para, residual_form=residual_sr
        )
        self.Estimator = Estimator(
            kernel_size=kernel_size, para_len=input_para, in_nc=in_nc, scale=self.scale, residual_form=residual_kernel
        )

        self.register_buffer("encoder", torch.load(pca_matrix_path)[None])

        kernel = torch.zeros(1, self.ksize, self.ksize)
        kernel[:, self.ksize // 2, self.ksize // 2] = 1

        self.register_buffer("init_kernel", kernel)
        init_ker_map = self.init_kernel.view(1, 1, self.ksize ** 2).matmul(
            self.encoder
        )[:, 0]
        self.register_buffer("init_ker_map", init_ker_map)

    def forward(self, lr):

        srs = []
        ker_maps = []
        kernels = []

        B, C, H, W = lr.shape
        ker_map = self.init_ker_map.repeat([B, 1])
        if self.residual_kernel:
            kernel = self.init_kernel.repeat([B, 1, 1]).view(B, self.ksize*self.ksize)

        if self.residual_sr:
            sr = torch.zeros((B, C, H*self.scale, W*self.scale)).to(lr.device)

        for i in range(self.loop):

            if self.residual_sr:
                sr = self.Restorer(lr, ker_map.detach(), previous_sr=sr.detach())
            else:
                sr = self.Restorer(lr, ker_map.detach())

            if self.residual_kernel:
                kernel = self.Estimator(sr.detach(), lr, previous_kernel=kernel.detach())
            else:
                kernel = self.Estimator(sr.detach(), lr)

            ker_map = kernel.view(B, 1, self.ksize ** 2).matmul(self.encoder)[:, 0]

            srs.append(sr)
            ker_maps.append(ker_map)
            kernels.append(kernel)

        return srs, ker_maps, kernels

