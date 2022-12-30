import torch
import torch.nn as nn
import torch.nn.functional as F
from .supmoco import SupMoCo


class WeakCon(SupMoCo):
    """
    Implementation of weak contrastive learning from https://doi.org/10.1016/j.knosys.2022.108984
    """

    def __init__(self, **kwargs):
        super(WeakCon, self).__init__(**kwargs)

        self.weight_error = nn.MSELoss(reduction='sum')

    def register_vector(self, vector_size):
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  # resets queue pointer
        self.register_buffer("queue_vectors", torch.zeros(vector_size, self.K).to(torch.float32).to(self.device))  # vectors for weight tracking

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, vectors):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        self.queue_vectors[:, ptr:ptr + batch_size] = vectors
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, q_vector=None, **kwargs):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            labels: Class label corresponding to each query/key image
        Output:
            logits, targets
        """
        if self.training:

            if q_vector is None:
                raise RuntimeError('Vector labels required for a training step.')

            batch_size = q_vector.size()[1]

            # compute query features
            embedding, q = self.encoder_q(im_q)  # queries: NxC
            q = nn.functional.normalize(q['q'], dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                _, k = self.encoder_k(im_k)  # keys: (N*P)xC
                k = nn.functional.normalize(k['q'], dim=1)

            # compute logits
            # positive logits: NxP
            l_pos = (torch.mul(q.unsqueeze(1),
                               k.reshape(im_q.shape[0], self.positives_per_class, self.vector_dim)))
            l_pos = ((l_pos.sum(dim=2)) / self.T).sum(dim=1)

            # divide by number of positives per class
            l_pos /= self.positives_per_class

            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # calculating weak contrastive weights:

            # option 1 - slow but the most obvious
            # weights = torch.ones(batch_size, self.K).to(self.device)
            # for b in range(batch_size):
            #     for i in range(self.queue.size()[1]):
            #         weights[b, i] = torch.sqrt(self.weight_error(q_vector[:, b], self.queue_vectors[:, i]))

            # option 2 - manual matching and tiling of vectors to allow simultaneous subtraction
            # q_expand = q_vector.unsqueeze(-1)
            # q_tile = torch.tile(q_expand, (1, 1, 8192))
            # queue_expand = self.queue_vectors.unsqueeze(0)
            # queue_expand_tile = torch.tile(queue_expand.transpose(1, 0), (1, 2, 1))
            # weights = torch.sqrt(torch.sum(nn.MSELoss(reduction='none')(q_tile, queue_expand_tile), 0))

            # option 3 - use pre-made pytorch function to compute result quickly (this is the selected method)
            weights = torch.cdist(q_vector.transpose(1, 0), self.queue_vectors.transpose(1, 0)).to(self.device)

            l_neg = l_neg * weights

            l_neg = l_neg/self.T

            # logits: Nx(1+K)
            logits = torch.cat([l_pos.unsqueeze(1), l_neg], dim=1)

            # labels: positive key indicators (always position 0)
            full_labels = torch.zeros(logits.shape[0], dtype=torch.long)

            # dequeue and enqueue (1 positive key image per query image)
            self._dequeue_and_enqueue(k[[i * self.positives_per_class for i in range(batch_size)]], q_vector)

            return embedding, logits, full_labels
        else:
            embedding, q = self.encoder_q(im_q)

            if 'get_q' in kwargs and kwargs['get_q']:
                return embedding, q['q']
            else:
                return embedding
