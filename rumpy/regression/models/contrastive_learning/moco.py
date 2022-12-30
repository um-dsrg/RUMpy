# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_models import AdjustedStandardModel


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    Two main changes were added to the original MoCo code:
    1) the ability to use more than one positive sample, and 2) the ability to use a variety of encoders.
    (Since MoCoV2 is better, all the models use an mlp, so the flag is unused for now)
    https://arxiv.org/abs/1911.05722
    https://github.com/facebookresearch/moco/blob/main/moco/builder.py
    """

    def __init__(self, base_encoder, dim=256, K=32*256, m=0.999, T=0.07, mlp=True, positives=1, dropdown=None):
        """
        dim: feature dimension (default: 256)
        K: queue size; number of negative keys (default: 8192)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.vector_dim = dim
        self.dropdown = dropdown
        self.positives = positives

        # create the encoders
        # num_classes is the output fc dimension
        if base_encoder.__name__ != 'Encoder' and base_encoder.__name__ != 'IDMN':
            # Use a standard model like ResNet
            # Pass it through AdjustedStandardModel to add an FC layer and get 2 outputs
            self.encoder_q = AdjustedStandardModel(base_encoder(num_classes=dim), dropdown_q=dropdown)
            self.encoder_k = AdjustedStandardModel(base_encoder(num_classes=dim), dropdown_q=dropdown)

            # NOTE: this is from the original code
            # if mlp:  # hack: brute-force replacement
            #     dim_mlp = self.encoder_q.fc.weight.shape[1]
            #     self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            #     self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
        else:
            # Use DASR encoder
            self.encoder_q = base_encoder(dropdown)
            self.encoder_k = base_encoder(dropdown)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, **kwargs):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if self.training:
            # compute query features
            embedding, q = self.encoder_q(im_q)  # queries: NxC
            q = nn.functional.normalize(q['q'], dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                _, k = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k['q'], dim=1)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            if self.positives == 1:
                l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            else:
                l_pos = (torch.mul(q.unsqueeze(1),
                         k.reshape(im_q.shape[0], self.positives, self.vector_dim)))

                l_pos = ((l_pos.sum(dim=2)) / self.T).sum(dim=1)

                # divide by number of positives
                l_pos /= self.positives

            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            if self.positives == 1:
                # logits: Nx(1+K)
                logits = torch.cat([l_pos, l_neg], dim=1)

                # apply temperature
                logits /= self.T
            else:
                l_neg /= self.T

                # logits: Nx(1+K)
                logits = torch.cat([l_pos.unsqueeze(1), l_neg], dim=1)

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long)

            # dequeue and enqueue
            if self.positives == 1:
                self._dequeue_and_enqueue(k)
            else:
                self._dequeue_and_enqueue(k[[i*self.positives for i in range(len(labels))]])

            return embedding, logits, labels
        else:
            embedding, q = self.encoder_q(im_q)

            if 'get_q' in kwargs and kwargs['get_q']:
                return embedding, q['q']
            else:
                return embedding


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

