import torch
from torch import nn
import numpy as np
import os
import csv

from rumpy.regression.models.contrastive_learning import BaseContrastive
from rumpy.regression.models.contrastive_learning.supmoco import SupMoCo
from rumpy.regression.models.contrastive_learning.moco import MoCo
from rumpy.regression.evaluation.eval_hub import ContrastiveEval, register_metadata, partition_metadata
from rumpy.regression.models.contrastive_learning.base_models import AdjustedStandardModel
from rumpy.SISR.models.blur_kernel_blind_sr.DCLS import DCLS


def load_encoder_model(weights, device, direct_load=False):
    if device == torch.device('cpu'):
        f_device = device
    else:
        f_device = "cuda:%d" % device
    state = torch.load(f=weights, map_location=f_device)
    if direct_load:
        return state
    encoder_dict = {}
    if state['model_name'] == 'mococontrastive' or state['model_name'] == 'supmoco' or state['model_name'] == 'weakcon':
        for key, val in state['network'].items():
            if 'encoder_q' in key:
                encoder_dict[key[10:]] = val
    elif state['model_name'] == 'supcon':
        encoder_dict = state['network']
    return encoder_dict


def setup_encoder(contrastive_encoder, encoder_freeze_mode, pre_trained_encoder_weights, device,
                  encoder_dropdown, load_required=False):
    # Encoder
    if contrastive_encoder == 'default':
        E = BaseContrastive.define_encoder_model(contrastive_encoder)(encoder_dropdown)
    elif contrastive_encoder == 'DCLS':
        E = DCLS(nb=10, input_para=256)
    else:
        E = AdjustedStandardModel(BaseContrastive.define_encoder_model(contrastive_encoder)(num_classes=256),
                                  encoder_dropdown)
        # TODO: num classes is hardcoded, do we ever change this?
    if encoder_freeze_mode == 'all':
        for param in E.parameters():
            param.requires_grad = False
    elif encoder_freeze_mode == 'pre_q':
        for name, param in E.named_parameters():
            if not 'mlp' in name:
                param.requires_grad = False

    if load_required:
        if contrastive_encoder == 'DCLS':
            encoder_dict = load_encoder_model(pre_trained_encoder_weights, device, direct_load=True)
        else:
            encoder_dict = load_encoder_model(pre_trained_encoder_weights, device)
        E.load_state_dict(state_dict=encoder_dict)
        print('Encoder weights loaded from %s' % pre_trained_encoder_weights)

    return E


class EncodingReducer(nn.Module):
    """
    Reduce degradation encoding before passing it to the model.
    """

    def __init__(self, reducer_layer_sizes=None):
        super(EncodingReducer, self).__init__()

        layers = []

        if reducer_layer_sizes:
            for i in range(len(reducer_layer_sizes) - 1):
                layers.append(nn.Conv2d(reducer_layer_sizes[i],
                                        reducer_layer_sizes[i + 1],
                                        1,
                                        padding=0,
                                        bias=True))
                layers.append(nn.ReLU(inplace=True))

        self.reducer = nn.Sequential(*layers)

    def forward(self, enc):
        out = self.reducer(enc)

        return out


class ContrastiveBlindSRPipeline(nn.Module):
    def __init__(self, device, eval_mode, generator,
                 contrastive_encoder='default', pre_trained_encoder_weights=None,
                 embedding_type='pre-q', encoder_freeze_mode='all',
                 auxiliary_encoder_weights=None, staggered_encoding=False,
                 encoding_normalization_type=None, encoding_normalization_params=None,
                 aux_encoding_normalization_params=None,
                 combined_loss_mode=None, crop_count=None,
                 checkpoint_load=False,
                 sft_mode=False,
                 srmd_mode=False,
                 contrastive_eval=False,
                 encoder_dropdown=None,
                 contrastive_dropdown=False,
                 reducer_layer_sizes=None,
                 block_encoder_loading=False,
                 **kwargs):
        super(ContrastiveBlindSRPipeline, self).__init__()

        if block_encoder_loading:
            # this is a parameter which blocks encoders from being loaded from file.
            # This is mostly only useful in a testing scenario.
            checkpoint_load = True

        self.combined_loss_mode = combined_loss_mode
        self.eval_mode = eval_mode
        self.staggered_encoding = staggered_encoding
        self.aux_E = None
        self.encoding_normalization_type = encoding_normalization_type
        self.encoding_normalization_params = encoding_normalization_params
        self.aux_encoding_normalization_params = aux_encoding_normalization_params
        self.device = device
        self.model_save_dir = kwargs['model_save_dir']

        if srmd_mode or sft_mode:
            self.sft_mode = True
        else:
            self.sft_mode = False

        self.srmd_mode = srmd_mode

        # Generator
        self.G = generator

        if embedding_type == 'pre-q':
            self.embed_digit = 0
            self.q_type = None
        elif embedding_type == 'q':  # q-version
            self.embed_digit = 1
            self.q_type = 'q'
        elif embedding_type == 'q-dropdown':
            self.embed_digit = 1
            self.q_type = 'dropdown_q'
        else:
            raise RuntimeError('Incorrect type of embedding selected.')

        if combined_loss_mode is None or combined_loss_mode == 'nonblind':
            # TODO: extend this function system to both types of loss...
            self.E = setup_encoder(contrastive_encoder, encoder_freeze_mode, pre_trained_encoder_weights,
                                   device, encoder_dropdown, load_required=not checkpoint_load)
            if auxiliary_encoder_weights is not None:
                if isinstance(auxiliary_encoder_weights, list):
                    self.aux_E = nn.ModuleList([setup_encoder(contrastive_encoder, encoder_freeze_mode, weights,
                                                              device, load_required=not checkpoint_load) for weights in
                                                auxiliary_encoder_weights])
                else:
                    self.aux_E = setup_encoder(contrastive_encoder, encoder_freeze_mode, auxiliary_encoder_weights,
                                               device, load_required=not checkpoint_load)

        else:
            if combined_loss_mode == 'moco':
                self.E = MoCo(base_encoder=BaseContrastive.define_encoder_model(contrastive_encoder), dropdown=encoder_dropdown)
            elif combined_loss_mode == 'supmoco':
                self.E = SupMoCo(device=device,
                                 base_encoder=BaseContrastive.define_encoder_model(contrastive_encoder),
                                 contrastive_dropdown=contrastive_dropdown,
                                 positives_per_class=crop_count - 1, dropdown=encoder_dropdown)

            if device == torch.device('cpu'):
                f_device = device
            else:
                f_device = "cuda:%d" % device

            if encoder_freeze_mode == 'all':
                for param in self.E.parameters():
                    param.requires_grad = False
            elif encoder_freeze_mode == 'pre_q':
                for name, param in self.E.named_parameters():
                    if not 'mlp' in name:
                        param.requires_grad = False

            if not checkpoint_load:
                state = torch.load(f=pre_trained_encoder_weights, map_location=f_device)

                encoder_dicts = []

                for encoder_name in ['encoder_q', 'encoder_k']:
                    encoder_dict = {}
                    for key, val in state['network'].items():
                        if encoder_name in key:
                            encoder_dict[key[10:]] = val
                    encoder_dicts.append(encoder_dict)

                self.E.encoder_q.load_state_dict(state_dict=encoder_dicts[0])
                self.E.encoder_k.load_state_dict(state_dict=encoder_dicts[1])
                self.E.queue = state['network']['queue']
                self.E.queue_labels = state['network']['queue_labels']
                self.E.queue_ptr = state['network']['queue_ptr']

        if contrastive_eval:
            self.eval_hub = ContrastiveEval()
            self.eval_hub.metadata_len = 12
            processed_keys = register_metadata(
                ['realesrganblur-sigma_x', 'realesrganblur-sigma_y', 'realesrganblur-rotation',
                 'realesrganblur-kernel_type', 'realesrganblur-beta_p', 'realesrganblur-beta_g',
                 'realesrganblur-omega_c', 'realesrganblur-kernel_size', 'downsample-scale',
                 'realesrgannoise-gaussian_noise_scale', 'realesrgannoise-gray_noise',
                 'realesrgannoise-poisson_noise_scale'])
            self.eval_hub.metadata_keys = processed_keys

            self.eval_hub.metadata_mapping = {key: processed_keys.index(key) for key in processed_keys}
            self.eval_hub.valid_metadata, self.eval_hub.decision_mags, self.eval_hub.total_classes = partition_metadata(
                self.eval_hub.metadata_mapping, labelling_strategy='triple_precision')
            self.eval_hub.config_output_plots(file_extension='png', dpi=100)
            self.eval_hub.register_hyperparams(
                '/opt/nfs/shared/scratch/Deep-FIR/Datasets_processed/div2k/lr_iso_blur_only')
            self.eval_hub.initialize_output_folder('/opt/users/maqui09/transfer_directory', 'contrastive_checkup')
            self.embedding_list = []
            self.q_list = []
            self.metadata_list = []

        self.contrastive_eval = contrastive_eval

        self.reducer = None

        if reducer_layer_sizes is not None:
            self.reducer = EncodingReducer(reducer_layer_sizes=reducer_layer_sizes)

    def normalize(self, vectors, norm_params):
        if self.encoding_normalization_type == 'minmax':
            maxi = norm_params['max']
            mini = norm_params['min']
            vectors = (vectors - mini) / (maxi - mini)
        elif self.encoding_normalization_type == 'meanstd':
            mean = norm_params['mean']
            std = norm_params['std']
            vectors = (vectors - mean) / std
        else:
            raise RuntimeError('Normalization type not recognized')
        return vectors

    def forward(self, x, x_key=None, labels=None, **kwargs):
        if self.combined_loss_mode is None or 'moco' not in self.combined_loss_mode:
            # Degradation representation (direct from encoder)
            embedding = self.E(x)[self.embed_digit]

            if isinstance(self.E, DCLS):
                embedding = embedding.squeeze(1).reshape((-1, 441))

            if self.q_type:
                embedding = embedding[self.q_type]

            if self.encoding_normalization_type is not None:
                embedding = self.normalize(embedding, self.encoding_normalization_params)

            if self.contrastive_eval:
                metadata = kwargs['metadata']
                q = self.E(x)[1]
                self.q_list.append(q.detach().cpu())
                self.embedding_list.append(embedding.detach().cpu())
                self.metadata_list.append(metadata)

                if len(self.embedding_list) == 40:
                    embedding_np = torch.vstack(self.embedding_list).squeeze().numpy()
                    q_np = torch.vstack(self.q_list).squeeze().numpy()
                    self.eval_hub.degradation_params = torch.vstack(self.metadata_list).squeeze().numpy()
                    self.eval_hub.data_encodings = embedding_np
                    self.eval_hub.data_q = q_np

                    self.eval_hub.interpret_metadata()
                    self.eval_hub.fit_tsne(normalize_fit=True, perplexity=40.0)

                    if 'gaussian_noise_scale' in self.eval_hub.metadata_keys:
                        self.eval_hub.plot_noise(plot_magnitudes=True, rep_type='tsne')

                    if 'jpeg_quality_factor' in self.eval_hub.metadata_keys:
                        self.eval_hub.plot_compression(rep_type='tsne')

                    if 'jpeg_quality_factor' in self.eval_hub.metadata_keys and 'gaussian_noise_scale' in self.eval_hub.metadata_keys:
                        self.eval_hub.plot_combined_noise_compression(rep_type='tsne')

                    if 'kernel_type' in self.eval_hub.metadata_keys:
                        self.eval_hub.plot_blur(rep_type='tsne')

            embedding = embedding.unsqueeze(2).unsqueeze(3)

            if self.reducer is not None:
                embedding = self.reducer(embedding)

            if self.aux_E:
                if isinstance(self.aux_E,
                              nn.ModuleList):  # TODO: staggered encoding for multiple aux embeddings not implemented
                    temp_embeddings = [aux(x)[self.embed_digit] for aux in self.aux_E]
                    aux_embedding = torch.cat((temp_embeddings[0], temp_embeddings[1]), 1)
                else:
                    aux_embedding = self.aux_E(x)[self.embed_digit]

                if self.encoding_normalization_type is not None:
                    aux_embedding = self.normalize(aux_embedding, self.aux_encoding_normalization_params)
                if self.staggered_encoding:
                    embedding = [embedding, aux_embedding.unsqueeze(2).unsqueeze(3)]
                else:
                    embedding = torch.cat((embedding.squeeze(-1).squeeze(-1), aux_embedding), 1).unsqueeze(2).unsqueeze(
                        3)

            # SR network (with image input and vector input)
            if self.sft_mode:
                upgraded_embedding = torch.ones(x.size(0), embedding.size(1), *x.size()[2:]).to(device=self.device)
                for batch in range(embedding.size(0)):
                    upgraded_embedding[batch, ...] = embedding[batch, ...].repeat_interleave(x.size()[2],
                                                                                             1).repeat_interleave(
                        x.size()[3], 2)
                if self.srmd_mode:
                    x_chan = torch.cat((x, upgraded_embedding), 1)
                    sr = self.G(x_chan, None)
                else:
                    sr = self.G(x, upgraded_embedding)
            else:
                sr = self.G(x, embedding)

            if self.reducer is not None and self.training:
                if self.combined_loss_mode is not None:
                    if self.combined_loss_mode == 'nonblind':
                        return sr, embedding

            if not self.training:
                norm_path = os.path.normpath(self.model_save_dir)
                split_path = norm_path.split(os.sep)

            return sr
        else:
            if self.training:
                if self.combined_loss_mode == 'moco':
                    embedding, logits, labels = self.E(x, x_key)
                elif self.combined_loss_mode == 'supmoco':
                    embedding, logits, labels, _ = self.E(x, x_key, labels)

                sr = self.G(x, embedding.unsqueeze(2).unsqueeze(3))
                return sr, logits, labels
            else:
                if self.combined_loss_mode == 'moco':
                    embedding = self.E(x, x_key, get_q=True)[self.embed_digit]
                elif self.combined_loss_mode == 'supmoco':
                    embedding = self.E(x, x_key, labels, get_q=True)[self.embed_digit]

                norm_path = os.path.normpath(self.model_save_dir)
                split_path = norm_path.split(os.sep)

                sr = self.G(x, embedding.unsqueeze(2).unsqueeze(3))

                return sr
