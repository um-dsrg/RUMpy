from collections import OrderedDict

from rumpy.SISR.models.advanced.architectures import common
from rumpy.SISR.models.attention_manipulators.da_layer import DA_conv
from rumpy.SISR.models.attention_manipulators.q_layer import ParaCALayer, ResPipesCALayer, ResPipesSplitCALayer
from rumpy.SISR.models.attention_manipulators.dgfmb_layer import DGFMBLayer
from rumpy.SISR.models.SFTMD_variants.architectures import StandardSft
from torch import nn


class MetaResBlock(nn.Module):
    """
    Res block similar to that of EDSR, but with additional metadata modulation capabilities.
    """
    def __init__(
            self, conv, n_feats, n_params, kernel_size, act=nn.ReLU(True), meta_type=None,
            dropout=False, dropout_probability=0.1,
            num_meta_layers=2, num_pipes=3, combine_pipes='concat', split_percent=0.25,
            bias=True, res_scale=1.0, sft_mask_para=False, use_linear=True):

        super(MetaResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)

        self.meta_type = meta_type
        if meta_type == 'q-layer':
            self.attention_layer = ParaCALayer(n_feats, n_params, num_layers=num_meta_layers, nonlinearity=True,
                                               dropout=dropout, dropout_probability=dropout_probability)
        elif meta_type == 'res-pipe-q-layer':
            self.attention_layer = ResPipesCALayer(n_feats, n_params, num_layers=num_meta_layers,
                                                   num_pipes=num_pipes, combine_pipes=combine_pipes, nonlinearity=True)
        elif meta_type == 'res-pipe-split-q-layer':
            self.attention_layer = ResPipesSplitCALayer(n_feats, n_params, num_layers=num_meta_layers,
                                                        num_pipes=num_pipes, split_percent=split_percent, nonlinearity=True)
        elif meta_type == 'SFT':
            self.attention_layer = StandardSft(nf=n_feats, para=n_params, mask_para=sft_mask_para)
        elif meta_type == 'da-layer':
            self.attention_layer = DA_conv(n_params, n_feats)
        elif meta_type == 'dgfmb-layer':
            self.attention_layer = DGFMBLayer(num_channels=n_feats, num_layers=num_meta_layers, use_linear=use_linear)

        self.res_scale = res_scale

    def forward(self, x):
        params = x[1]
        res = self.body(x[0])
        res = res.mul(self.res_scale)
        if self.meta_type is not None:
            if self.meta_type == 'da-layer':
                res = self.attention_layer((res, x[1]))
            else:
                res = self.attention_layer(res, x[1])
        res += x[0]

        return res, params

    def forensic(self, x):
        params = x[1]
        res = self.body(x[0])
        res = res.mul(self.res_scale)

        forensic_data = OrderedDict()

        if self.meta_type is not None:
            if self.meta_type == 'da-layer':
                res = self.attention_layer((res, x[1]))

                unchanged_features = self.attention_layer.forensic((res, x[1]))[1]
                multiplied_features = self.attention_layer.forensic((res, x[1]))[2]
                attention_only = self.attention_layer.forensic((res, x[1]))[3]

                forensic_data['unchanged_features'] = unchanged_features
                forensic_data['multiplied_features'] = multiplied_features
                forensic_data['attention_vector'] = attention_only
            else:
                res = self.attention_layer(res, x[1])

                unchanged_features = self.attention_layer.forensic(res, x[1])[1]
                multiplied_features = self.attention_layer.forensic(res, x[1])[2]
                attention_only = self.attention_layer.forensic(res, x[1])[3]

                forensic_data['unchanged_features'] = unchanged_features
                forensic_data['multiplied_features'] = multiplied_features
                forensic_data['attention_vector'] = attention_only
        else:
            forensic_data = None

        res += x[0]

        return res, params, forensic_data


class MetadataEncoder(nn.Module):
    """
    Encoder to transform blurring, gender, age, etc. metadata into a continuous vector.
    """
    def __init__(self,
                 input_para=1, num_bottleneck_nodes=16, encoder_layers_sizes=None):
        super(MetadataEncoder, self).__init__()

        layers = []

        all_layer_sizes = [input_para] + encoder_layers_sizes + [num_bottleneck_nodes]

        if encoder_layers_sizes:
            for i in range(len(all_layer_sizes) - 1):
                layers.append(nn.Conv2d(all_layer_sizes[i], all_layer_sizes[i+1],
                                        1, padding=0, bias=True))
                layers.append(nn.ReLU(inplace=True))
        else:
            nn.Conv2d(input_para, 36, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 24, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, num_bottleneck_nodes, 1, padding=0, bias=True),
            nn.ReLU(inplace=True)

        self.encoder = nn.Sequential(*layers)

    def forward(self, metadata):
        enc = self.encoder(metadata)

        return enc

class MetadataDecoder(nn.Module):
    """
    Decoder to transform the encoded vector back into the metadata.
    """
    def __init__(self,
                 output_para=1, num_bottleneck_nodes=16, decoder_layers_sizes=None):
        super(MetadataDecoder, self).__init__()

        layers = []

        all_layer_sizes = [num_bottleneck_nodes] + decoder_layers_sizes + [output_para]

        if decoder_layers_sizes:
            for i in range(len(all_layer_sizes) - 1):
                layers.append(nn.Conv2d(all_layer_sizes[i], all_layer_sizes[i+1],
                                        1, padding=0, bias=True))
                layers.append(nn.ReLU(inplace=True))
        else:
            layers = [
                nn.Conv2d(num_bottleneck_nodes, 24, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 36, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(36, output_para, 1, padding=0, bias=True),
                nn.ReLU(inplace=True)
            ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, enc):
        out = self.decoder(enc)

        return out


class Metabed(nn.Module):
    """
    Testbed for comparing metadata insertion systems.  Base model is a truncated EDSR which allows for
    quick training turnaround.
    """
    def __init__(self,
                 in_features=3, out_features=3, num_features=64, input_para=1, meta_block=None, num_meta_layers=2,
                 num_pipes=3, combine_pipes='concat', split_percent=0.25,
                 use_encoder=None, num_bottleneck_nodes=16, encoder_layers_sizes=None, decoder_layers_sizes=None,
                 selective_meta_blocks=None, num_blocks=1, scale=4, res_scale=0.1, sft_mask_para=None, use_linear=True, **kwargs):
        """
        :param in_features: Input image channels (3 for RGB).
        :param out_features: Output image channels (3 for RGB).
        :param num_features: Network feature map channels.
        :param input_para: Metadata input vector size.
        :param meta_block: One of 'q-layer' (meta-attention), 'da-layer'
        (Degradation-aware block, taken from DASR: https://arxiv.org/pdf/2104.00416.pdf), 'SFT'
        (spatial feature transform - taken from SFTMD (https://arxiv.org/pdf/1904.03377.pdf),
        res-pipe-q-layer (meta-attention) or res-pipe-split-q-layer (meta-attention).
        :param num_meta_layers: Number of meta-layers to add into network or list of layers (in case of res-pipe).
        :param num_pipes: Number of pipe (or paths) that the metadata will flow through (in case of res-pipes or res-pipes-split).
        :param combine_pipes: Method of combining outputs from pipes (or paths). Can be 'add' or 'concat' (only in case of res-pipes).
        :param split_percent: Fraction showing the percentage of elements that each pipe will output (only in case of res-pipes-split).
        E.g. If num_features = 64 and split_percent = 0.25; 16 (64*0.25) features will be sent to the concat layer while the remaining
        48 (64 - 16) will be sent to the next pipe.
        :param selective_meta_blocks: List of Bools; must be the same length as the number of residual blocks.
        Setting an element of the list to True will signal the addition of a meta-layer
        in the corresponding residual block.
        :param num_blocks: Number of residual blocks.
        :param scale: SR scale.
        :param res_scale: Residual magnitude scaling factor.
        :param sft_mask_para:  Set to True to add SFT blocks without any metadata.
        :param use_linear: Set to True to use Linear Fully Connected layers, False to use Conv2d layers. (So far only for DGFMB.)
        """
        super(Metabed, self).__init__()

        n_resblocks = num_blocks
        n_feats = num_features
        kernel_size = 3

        self.use_encoder = use_encoder

        metadata_size = input_para

        if use_encoder:
            self.meta_enc = MetadataEncoder(input_para=input_para,
                                            num_bottleneck_nodes=num_bottleneck_nodes,
                                            encoder_layers_sizes=encoder_layers_sizes)
            self.meta_dec = MetadataDecoder(output_para=input_para,
                                            num_bottleneck_nodes=num_bottleneck_nodes,
                                            decoder_layers_sizes=decoder_layers_sizes)
            metadata_size = num_bottleneck_nodes

        # define head module
        self.head = common.default_conv(in_features, n_feats, kernel_size)

        # define body module
        m_body = []
        if selective_meta_blocks is None:
            for index in range(n_resblocks):
                m_body.append(MetaResBlock(
                    common.default_conv, n_feats, metadata_size, kernel_size, res_scale=res_scale, meta_type=meta_block,
                    num_meta_layers=num_meta_layers, num_pipes=num_pipes,
                    combine_pipes=combine_pipes, split_percent=split_percent,
                    sft_mask_para=False if sft_mask_para is None else sft_mask_para[index], use_linear=use_linear))
        else:
            for index in range(n_resblocks):
                if selective_meta_blocks[index]:
                    meta_insert = meta_block
                    if sft_mask_para is not None:
                        sft_mask = sft_mask_para[index]
                    else:
                        sft_mask = False
                else:
                    meta_insert = None
                    sft_mask = False
                m_body.append(MetaResBlock(
                    common.default_conv, n_feats, metadata_size, kernel_size, res_scale=res_scale, sft_mask_para=sft_mask,
                    num_meta_layers=num_meta_layers, num_pipes=num_pipes, combine_pipes=combine_pipes,
                    split_percent=split_percent, meta_type=meta_insert, use_linear=use_linear))

        self.final_body = common.default_conv(n_feats, n_feats, kernel_size)

        # define tail module
        m_tail = [
            common.Upsampler(common.default_conv, scale, n_feats),
            common.default_conv(n_feats, out_features, kernel_size)
        ]

        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, metadata):
        x = self.head(x)

        res, _ = self.body((x, metadata))

        res = self.final_body(res)
        res += x
        x = self.tail(res)

        return x

    def forensic(self, x, metadata):
        x = self.head(x)

        forensic_data = OrderedDict()

        for index, module in enumerate(self.body):
            if isinstance(module, MetaResBlock):
                res, _, res_forensic_data = module.forensic((x, metadata))
                if res_forensic_data is not None:
                    forensic_data['R%d' % (index)] = res_forensic_data
            else:
                res, _ = module.forward((x, metadata))

        res = self.final_body(res)
        res += x
        x = self.tail(res)
        return x, forensic_data
