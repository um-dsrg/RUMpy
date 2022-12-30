# all code originally obtained from https://github.com/greatlog/DAN
import torch
import torch.nn as nn
import os
from rumpy.SISR.models.blur_kernel_blind_sr.DANv1 import Estimator
from rumpy.SISR.models.attention_manipulators.architectures import QRCAN, QHAN, QELAN, QRRDBNet


class DANv1QRCAN(nn.Module):
    def __init__(
            self,
            nf=64,
            upscale=4,
            input_para=10,
            kernel_size=21,
            loop=4,
            pca_matrix_path=os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir)), 'pca_matrix.pth'),
            use_pca_encoder=True,
            **kwargs
    ):
        super(DANv1QRCAN, self).__init__()

        self.ksize = kernel_size
        self.loop = loop
        self.scale = upscale

        self.Restorer = QRCAN(n_feats=nf, scale=upscale, num_metadata=input_para, **kwargs)
        self.Estimator = Estimator(out_nc=input_para, kernel_size=kernel_size, scale=self.scale)

        self.register_buffer("encoder", torch.load(pca_matrix_path)[None])

        if use_pca_encoder:
            kernel = torch.zeros(1, self.ksize, self.ksize)
            kernel[:, self.ksize // 2, self.ksize // 2] = 1

            self.register_buffer("init_kernel", kernel)
            init_ker_map = self.init_kernel.view(1, 1, self.ksize ** 2).matmul(
                self.encoder
            )[:, 0]
        else:
            init_ker_map = torch.full((1, input_para), 0.5)

        self.register_buffer("init_ker_map", init_ker_map)

    def forward(self, lr=None, train=False):
        if lr is None:
            return
        srs = []
        ker_maps = []

        B, C, H, W = lr.shape
        ker_map = self.init_ker_map.repeat([B, 1])

        for i in range(self.loop):

            sr = self.Restorer(lr, ker_map.unsqueeze(-1).unsqueeze(-1).detach())
            ker_map = self.Estimator(sr.detach(), lr)

            srs.append(sr)
            ker_maps.append(ker_map)

        return srs, ker_maps


class DANv1QHAN(nn.Module):
    def __init__(
            self,
            nf=64,
            upscale=4,
            input_para=10,
            kernel_size=21,
            loop=4,
            pca_matrix_path=os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir)), 'pca_matrix.pth'),
            use_pca_encoder=True,
            **kwargs
    ):
        super(DANv1QHAN, self).__init__()

        self.ksize = kernel_size
        self.loop = loop
        self.scale = upscale

        self.Restorer = QHAN(n_feats=nf, scale=upscale, num_metadata=input_para, **kwargs)
        self.Estimator = Estimator(out_nc=input_para, kernel_size=kernel_size, scale=self.scale)

        self.register_buffer("encoder", torch.load(pca_matrix_path)[None])

        if use_pca_encoder:
            kernel = torch.zeros(1, self.ksize, self.ksize)
            kernel[:, self.ksize // 2, self.ksize // 2] = 1

            self.register_buffer("init_kernel", kernel)
            init_ker_map = self.init_kernel.view(1, 1, self.ksize ** 2).matmul(
                self.encoder
            )[:, 0]
        else:
            init_ker_map = torch.full((1, input_para), 0.5)

        self.register_buffer("init_ker_map", init_ker_map)

    def forward(self, lr=None, train=False):
        if lr is None:
            return
        srs = []
        ker_maps = []

        B, C, H, W = lr.shape
        ker_map = self.init_ker_map.repeat([B, 1])

        for i in range(self.loop):

            sr = self.Restorer(lr, ker_map.unsqueeze(-1).unsqueeze(-1).detach())
            ker_map = self.Estimator(sr.detach(), lr)

            srs.append(sr)
            ker_maps.append(ker_map)

        return srs, ker_maps


class DANv1QELAN(nn.Module):
    def __init__(
            self,
            nf=64,
            upscale=4,
            input_para=10,
            kernel_size=21,
            loop=4,
            pca_matrix_path=os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir)), 'pca_matrix.pth'),
            use_pca_encoder=True,
            **kwargs
    ):
        super(DANv1QELAN, self).__init__()

        self.ksize = kernel_size
        self.loop = loop
        self.scale = upscale

        self.Restorer = QELAN(n_feats=nf, scale=upscale, num_metadata=input_para, **kwargs)
        self.Estimator = Estimator(out_nc=input_para, kernel_size=kernel_size, scale=self.scale)

        self.register_buffer("encoder", torch.load(pca_matrix_path)[None])

        if use_pca_encoder:
            kernel = torch.zeros(1, self.ksize, self.ksize)
            kernel[:, self.ksize // 2, self.ksize // 2] = 1

            self.register_buffer("init_kernel", kernel)
            init_ker_map = self.init_kernel.view(1, 1, self.ksize ** 2).matmul(
                self.encoder
            )[:, 0]
        else:
            init_ker_map = torch.full((1, input_para), 0.5)

        self.register_buffer("init_ker_map", init_ker_map)

    def forward(self, lr=None, train=False):
        if lr is None:
            return
        srs = []
        ker_maps = []

        B, C, H, W = lr.shape
        ker_map = self.init_ker_map.repeat([B, 1])

        for i in range(self.loop):

            sr = self.Restorer(lr, ker_map.unsqueeze(-1).unsqueeze(-1).detach())
            ker_map = self.Estimator(sr.detach(), lr)

            srs.append(sr)
            ker_maps.append(ker_map)

        return srs, ker_maps

class DANv1QRRDB(nn.Module):
    def __init__(
            self,
            nf=64,
            upscale=4,
            input_para=10,
            kernel_size=21,
            loop=4,
            pca_matrix_path=os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir)), 'pca_matrix.pth'),
            use_pca_encoder=True,
            **kwargs
    ):
        super(DANv1QRRDB, self).__init__()

        self.ksize = kernel_size
        self.loop = loop
        self.scale = upscale

        self.Restorer = QRRDBNet(num_feat=nf, scale=upscale, num_metadata=input_para, **kwargs)
        self.Estimator = Estimator(out_nc=input_para, kernel_size=kernel_size, scale=self.scale)

        self.register_buffer("encoder", torch.load(pca_matrix_path)[None])

        if use_pca_encoder:
            kernel = torch.zeros(1, self.ksize, self.ksize)
            kernel[:, self.ksize // 2, self.ksize // 2] = 1

            self.register_buffer("init_kernel", kernel)
            init_ker_map = self.init_kernel.view(1, 1, self.ksize ** 2).matmul(
                self.encoder
            )[:, 0]
        else:
            init_ker_map = torch.full((1, input_para), 0.5)

        self.register_buffer("init_ker_map", init_ker_map)

    def forward(self, lr=None, train=False):
        if lr is None:
            return
        srs = []
        ker_maps = []

        B, C, H, W = lr.shape
        ker_map = self.init_ker_map.repeat([B, 1])

        for i in range(self.loop):

            sr = self.Restorer(lr, ker_map.unsqueeze(-1).unsqueeze(-1).detach())
            ker_map = self.Estimator(sr.detach(), lr)

            srs.append(sr)
            ker_maps.append(ker_map)

        return srs, ker_maps
