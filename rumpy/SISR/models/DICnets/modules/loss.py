import torch
import torch.nn as nn
import torch.nn.functional as F

# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, \
            grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss

class StructureLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat_list_1, feat_list_2, index_list):
        assert len(feat_list_1) == len(feat_list_2)
        loss = 0.0
        for i in range(len(feat_list_1)):
            feat1 = feat_list_1[i]
            feat2 = feat_list_2[i]
            index = index_list[i]
            struc_vec1 = calc_struc_vec(feat1, index)
            struc_vec2 = calc_struc_vec(feat2, index)
            loss += (struc_vec1 - struc_vec2).abs().mean()
        loss /= len(feat_list_1)
        # loss in between [0, 2]
        return loss

def calc_struc_vec(feat, index):
    '''
    @param
    feat: N * C * H * W
    index: N * num_anchor * 9 * 2, [:, :, 0, :] is center point
        [:, :, 1:, :] are surrounding points
    @return
    struc_vec: N * num_anchor * 8, sturcture vector
    '''
    assert feat.size(0) == index.size(0)
    bsize = feat.size(0)
    num_anchor = index.size(1)
    num_c = feat.size(1)
    # pad feat and index
    feat = F.pad(feat, (1, 1, 1, 1), mode='reflect')
    h, w = feat.shape[-2:]
    pad_index = index + 1
    
    # select feature vector
    index_x = pad_index[:, :, :, 0].view(bsize, 1, num_anchor * 9, 1).repeat(1, num_c, 1, h)
    index_y = pad_index[:, :, :, 1].view(bsize, 1, num_anchor * 9, 1).repeat(1, num_c, 1, 1)
    feat_x = feat.gather(-2, index_x)
    feat_selected = feat_x.gather(-1, index_y).squeeze(-1).squeeze(-1)
    feat_selected = feat_selected.transpose(1, 2).contiguous().view(bsize, num_anchor, 9, num_c)
    feat_selected = feat_selected
    
    # calculate dot product
    round_vec = feat_selected.view(bsize * num_anchor, 9, num_c)[:, 1:, :]
    center_vec = feat_selected.view(bsize * num_anchor, 9, num_c)[:, [0], :]
    struc_vec = torch.bmm(round_vec, center_vec.transpose(1, 2))
    norm = round_vec.pow(2).sum(-1, keepdim=True).sqrt() * center_vec.pow(2).sum(-1, keepdim=True).sqrt()
    struc_vec /= norm
    struc_vec = struc_vec.view(bsize, num_anchor, 8)
    
    return struc_vec
