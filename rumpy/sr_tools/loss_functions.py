from torch import nn
import torch
from rumpy.SISR.models.feature_extractors.handlers import perceptual_loss_mechanism


class OccupancyLoss(nn.Module):
    def __init__(self, device, zero_thres=1e-6):
        super(OccupancyLoss, self).__init__()
        self.pixel_loss = nn.L1Loss()
        self.zero_thres = zero_thres
        self.device = device

    def forward(self, pred, gt):
        thres_gt = torch.where(gt > self.zero_thres, 1, 0).to(self.device)
        thres_pred = torch.where(pred > self.zero_thres, 1, 0).to(self.device)
        gt_occ = (gt/gt) * thres_gt
        pred_occ = (pred/pred) * thres_pred

        return torch.sum(torch.abs(gt_occ - pred_occ))


class PerceptualMechanism(nn.Module):
    def __init__(self, device=torch.device('cpu'), lambda_pixel=1, lambda_per=0.01, vgg_type='vgg', vgg_mode='p_loss'):
        super(PerceptualMechanism, self).__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_per = lambda_per
        self.vgg_extractor = perceptual_loss_mechanism(vgg_type, mode=vgg_mode, device=device)
        self.vgg_extractor.to(device)
        self.vgg_extractor.eval()
        self.vgg_loss = nn.L1Loss()
        self.pixel_loss = nn.L1Loss()
        self.device = device

    def forward(self, sr, y):
        gen_features = self.vgg_extractor(sr)
        real_features = self.vgg_extractor(y).detach()
        vgg_loss = self.vgg_loss(gen_features, real_features)
        return (self.lambda_pixel*self.pixel_loss(sr, y)) + (self.lambda_per*vgg_loss)


class SupConLoss(nn.Module):
    """This code was extracted from https://github.com/HobbitLong/SupContrast

    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



# # Additional parameters compared to MoCo
# # queue_y: a new queue to store labels (K,)
# # y: labels for query images
# # P: number of positives per class
# # T : total number of classes (0 .. T-1)
# # initialize
#
# f_k.params = f_q.params
# queue_y.fill_(T)
# for x in loader:  # load a minibatch x with N samples
#     x_q = aug(x)  # a randomly augmented version
#     x_k = aug(x)  # P positives per image
#     q = f_q.forward(x_q)  # queries: NxC
#     k = f_k.forward(x_k)  # keys: NxC
#     k = k.detach()  # no gradient to keys
#     # positive logits from batch: N x P
#     l_pos = (torch.mul(q.unsqueeze(1),
#                        k.reshape(N, P, C)))
#     l_pos = (l_pos.sum(dim=2)) / t
#     # labels from queue: N X K,
#     # each value of K indicates positive or not
#     yb = torch.nn.functional.one_hot(y, T + 1)
#     yq = torch.nn.functional.one_hot(queue_y, T + 1)
#     pos_y_q = torch.matmul(yb, yq.t())
#     # sum of all positive features from queue: N X C
#     pos_f_q = torch.matmul(pos_y_q, queue.t())
#     # compute cosine similarity with q : N X 1
#     pos_q = (torch.mul(q, pos_f_q) / t).sum(dim=1)
#     # Number of positives for each x_q : N X 1
#     num_positives = P + pos_y_q.sum(dim=1)
#     # Combine batch and queue positives: N X 1
#     l_pos = l_pos.sum(dim=1) + pos_q
#     # divide by number of positives per class
#     l_pos /= num_positives
#     # negative logits computation stays the same
#     l_neg = torch.matmul(q, queue) / t
#     # Compute contrastive loss (Eq. 3) and update parameters
#     # Enqueue and dequeue images and labels, 1 per P positives
