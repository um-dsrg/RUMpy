from torch import nn
from rumpy.SISR.models.advanced.SAN_blocks import SOCA, RB
from rumpy.SISR.models.attention_manipulators.q_layer import ParaCALayer
import torch


# meta-enhanced Residual  Block (QRB)
class QRB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(inplace=True),
                 res_scale=1, dilation=2, num_metadata=0):
        super(QRB, self).__init__()
        modules_body = []

        # self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma1 = 1.0
        # self.salayer = SALayer(n_feat, reduction=reduction, dilation=dilation)
        # self.salayer = SALayer2(n_feat, reduction=reduction, dilation=dilation)


        self.conv_first = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias),
                                        act,
                                        conv(n_feat, n_feat, kernel_size, bias=bias)
                                        )

        self.res_scale = res_scale
        self.q_layer = ParaCALayer(n_feat, num_metadata, nonlinearity=True, num_layers=2)

    def forward(self, x):
        y = self.conv_first(x[0])
        y = self.q_layer(y, x[1])
        y = y + x[0]

        return y


## metadata-enhanced Local-source Residual Attention Group (LSRARG)
class QLSRAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks,
                 num_metadata=0, num_q_layers=None):
        super(QLSRAG, self).__init__()

        if num_q_layers is None:
            self.rcab = nn.ModuleList([QRB(conv, n_feat, kernel_size, reduction,
                                           bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1,
                                           num_metadata=num_metadata) for _ in range(n_resblocks)])
        else:
            modules = []
            for i in range(n_resblocks):
                if i < num_q_layers:
                    modules.append(QRB(conv, n_feat, kernel_size, reduction,
                                           bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1,
                                           num_metadata=num_metadata))
                else:
                    modules.append(RB(conv, n_feat, kernel_size, reduction,
                                       bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1))
            self.rcab = nn.ModuleList(modules)

        self.soca = (SOCA(n_feat, reduction=reduction))
        self.conv_last = (conv(n_feat, n_feat, kernel_size))
        self.n_resblocks = n_resblocks
        ##
        # modules_body = []
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = 0.2
        # for i in range(n_resblocks):
        #     modules_body.append(RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1))
        # modules_body.append(SOCA(n_feat,reduction=reduction))
        # # modules_body.append(Nonlocal_CA(in_feat=n_feat, inter_feat=n_feat//8, reduction =reduction, sub_sample=False, bn_layer=False))
        # modules_body.append(conv(n_feat, n_feat, kernel_size))
        # self.body = nn.Sequential(*modules_body)
        ##

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.ModuleList(layers)
        # return nn.Sequential(*layers)

    def forward(self, x):
        params = x[1]
        # batch_size,C,H,W = x.shape
        # y_pre = self.body(x)
        # y_pre = y_pre + x
        # return y_pre

        ## share-source skip connection
        flow_through = x[0]

        for i, l in enumerate(self.rcab):
            # x = l(x) + self.gamma*residual
            if isinstance(l, RB):
                flow_through = l(flow_through)
            else:
                flow_through = l((flow_through, params))
        flow_through = self.soca(flow_through)
        flow_through = self.conv_last(flow_through)

        flow_through = x[0] + flow_through

        return flow_through, params
        ##
