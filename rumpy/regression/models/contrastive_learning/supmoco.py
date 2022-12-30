import torch
import torch.nn as nn
import torch.nn.functional as F
from rumpy.regression.models.contrastive_learning.moco import MoCo


class SupMoCo(MoCo):
    """
    Implementation of SupMoCo from https://arxiv.org/pdf/2101.11058.pdf
    """

    def __init__(self, device, positives_per_class=4, contrastive_dropdown=True, **kwargs):
        """
        dim: feature dimension (default: 256).
        K: queue size; number of negative keys (default: 8192).
        m: moco momentum of updating key encoder (default: 0.999).
        T: softmax temperature (default: 0.07).
        num_classes: Number of possible object classes.
        positives_per_class: Number of positive samples provided per class in a batch.
        """
        super(SupMoCo, self).__init__(**kwargs)

        self.num_classes = 0
        self.positives_per_class = positives_per_class
        self.device = device
        self.contrastive_dropdown = contrastive_dropdown

    def register_classes(self, num_classes):
        self.set_class_count(num_classes)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  # resets queue pointer
        self.register_buffer("queue_labels", (torch.ones(self.K) * num_classes).to(torch.int64).to(
            self.device))  # labels for +ve tracking

    def set_class_count(self, num_classes):
        self.num_classes = num_classes

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        self.queue_labels[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, labels=None, **kwargs):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            labels: Class label corresponding to each query/key image
        Output:
            logits, targets
        """
        if self.training:
            if self.num_classes == 0:
                raise RuntimeError('Maximum number of classes must be registered before running a training step.')

            if labels is None:
                raise RuntimeError('Labels required for a training step.')

            embedding, final_outputs = self.encoder_q(im_q)  # queries: NxC
            # compute query features
            if self.dropdown and self.contrastive_dropdown:  # TODO: less hardcoding required here
                q = final_outputs['dropdown_q']
            else:
                q = final_outputs['q']

            q = nn.functional.normalize(q, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                _, k_outputs = self.encoder_k(im_k)  # keys: (N*P)xC
                if self.dropdown and self.contrastive_dropdown:
                    k = k_outputs['dropdown_q']
                else:
                    k = k_outputs['q']
                k = nn.functional.normalize(k, dim=1)

            # compute logits
            # positive logits: NxP
            l_pos = (torch.mul(q.unsqueeze(1),
                               k.reshape(im_q.shape[0], self.positives_per_class, self.vector_dim)))
            l_pos = (l_pos.sum(dim=2)) / self.T

            # labels from queue: N X K,
            # each value of K indicates positive or not
            yb = torch.nn.functional.one_hot(labels.to(torch.int64), int(self.num_classes) + 1).to(torch.float32)
            yq = torch.nn.functional.one_hot(self.queue_labels, int(self.num_classes) + 1).to(torch.float32)
            pos_y_q = torch.matmul(yb, yq.t())

            # sum of all positive features from queue: N X C
            pos_f_q = torch.matmul(pos_y_q, self.queue.t())

            # compute cosine similarity with q : N X 1
            pos_q = (torch.mul(q, pos_f_q) / self.T).sum(dim=1)

            # Number of positives for each x_q : N X 1
            num_positives = self.positives_per_class + pos_y_q.sum(dim=1)

            # Combine batch and queue positives: N X 1
            l_pos = l_pos.sum(dim=1) + pos_q

            # divide by number of positives per class
            l_pos /= num_positives

            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()]) / self.T

            # logits: Nx(1+K)
            logits = torch.cat([l_pos.unsqueeze(1), l_neg], dim=1)

            # labels: positive key indicators (always position 0)
            full_labels = torch.zeros(logits.shape[0], dtype=torch.long)

            # dequeue and enqueue (1 positive key image per query image)
            self._dequeue_and_enqueue(k[[i * self.positives_per_class for i in range(len(labels))]], labels)

            return embedding, logits, full_labels, final_outputs  # TODO: need to handle this better
        else:
            embedding, q_out = self.encoder_q(im_q)  # TODO: what if dropdown_q is requested?

            if 'get_q' in kwargs and kwargs['get_q']:
                if self.dropdown:
                    return embedding, q_out
                else:
                    return embedding, q_out['q']
            else:
                return embedding

