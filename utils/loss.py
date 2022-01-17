import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def cross_entropy_2d(pred,label,cfg,device):
    '''
    Args:
        predict:(n, c, h, w)
        target: (n, h, w)
    '''
    assert not label.requires_grad
    assert pred.dim()   == 4
    assert label.dim()  == 3
    assert pred.size(0) == label.size(0), f'{pred.size(0)}vs{label.size(0)}'
    assert pred.size(2) == label.size(1), f'{pred.size(2)}vs{label.size(2)}'
    assert pred.size(3) == label.size(2), f'{pred.size(3)}vs{label.size(3)}'

    n,c,h,w = pred.size()
    label   = label.view(-1)
    class_count = torch.bincount(label).float()
    try:
        assert class_count.size(0) == 5
        new_class_count = class_count
    except:
        new_class_count = torch.zeros(5).cuda().float()
        new_class_count[:class_count.size(0)] = class_count


    weight      = (1 - new_class_count/label.size(0))
    pred    = pred.transpose(1,2).transpose(2,3).contiguous() #n*c*h*w->n*h*c*w->n*h*w*c
    pred    = pred.view(-1,c)
    loss    = F.cross_entropy(input=pred,target=label,weight=weight)

    return loss



def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))

def dice_loss(pred, target, device):
    """
    input is a torch variable of size [N,C,H,W]
    target: [N,H,W]
    """
    n,c,h,w        = pred.size()
    pred           = pred.cuda()
    target         = target.cuda()
    target_onehot  = torch.zeros([n,c,h,w]).cuda()
    target         = torch.unsqueeze(target,dim=1) # n*1*h*w
    target_onehot.scatter_(1,target,1)

    assert pred.size() == target_onehot.size(), "Input sizes must be equal."
    assert pred.dim()  == 4, "Input must be a 4D Tensor."
    uniques = np.unique(target_onehot.cpu().data.numpy())
    assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    eps = 1e-5
    probs = F.softmax(pred,dim=1)
    num   = probs * target_onehot  # b,c,h,w--p*g
    num   = torch.sum(num, dim=3)  # b,c,h
    num   = torch.sum(num, dim=2)  # b,c,

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)  # b,c,h
    den1 = torch.sum(den1, dim=2)  # b,c,

    den2 = target_onehot * target_onehot  # --g^2
    den2 = torch.sum(den2, dim=3)  # b,c,h
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2.0 * (num / (den1 + den2+eps))  # b,c

    dice_total =  torch.sum(dice) / dice.size(0)  # divide by batch_sz
    return 1 - 1.0 * dice_total/5.0



class MPCL(nn.Module):
    def __init__(self,device, num_class=5,temperature=0.07,m=0.5,
                 base_temperature=0.07,easy_margin=False):
        super(MPCL, self).__init__()
        self.num_class        = num_class
        self.temperature      = temperature
        self.base_temperature = base_temperature
        self.m                = m
        self.cos_m            = math.cos(m)
        self.sin_m            = math.sin(m)
        self.th               = math.cos(math.pi - m)
        self.mm               = math.sin(math.pi - m) * m
        self.device           = device
        self.easy_margin      = easy_margin

    def forward(self, features, labels,class_center_feas,
                pixel_sel_loc=None, mask=None):
        """

         features: [batch_size*fea_h*fea_w] * 1 *c  normalized
         labels:   batch_size*fea_h*fea_w
         class_center_feas:  n_fea*n_class  normalized

        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        # build mask
        num_samples = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(num_samples, dtype=torch.float32).cuda()
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1).long()  # n_sample*1
            class_center_labels = torch.range(0,self.num_class-1).long().cuda()
            # print(class_center_labels)
            class_center_labels = class_center_labels.contiguous().view(-1,1) # n_class*1
            if labels.shape[0] != num_samples:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels,torch.transpose(class_center_labels,0,1)).float().cuda() # broadcast n_sample*n_class
        else:
            mask = mask.float().cuda()
        # n_sample = batch_size * fea_h * fea_w
        # mask n_sample*n_class  the mask_ij represents whether the i-th sample has the same label with j-th class or not.
        # in our experiment, the n_view = 1, so the contrast_count = 1
        contrast_count   = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [n*h*w]*fea_s

        anchor_feature = contrast_feature
        anchor_count   = contrast_count


        # compute logits
        cosine = torch.matmul(anchor_feature, class_center_feas) # [n*h*w] * n_class
        logits = torch.div(cosine,self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits        = logits - logits_max.detach()

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0.0001, 1.0))
        phi  = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # print(phi)
        phi_logits = torch.div(phi,self.temperature)

        phi_logits_max, _ = torch.max(phi_logits, dim=1, keepdim=True)
        phi_logits = phi_logits - phi_logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)

        tag_1             = (1-mask)
        tag_2             = mask
        exp_logits        = torch.exp(logits*tag_1 + phi_logits * tag_2)
        phi_logits        = (logits*tag_1) + (phi_logits*tag_2)
        log_prob          = phi_logits - torch.log(exp_logits.sum(1, keepdim=True)+1e-4)


        if pixel_sel_loc is not None:

            pixel_sel_loc     = pixel_sel_loc.view(-1)

            mean_log_prob_pos =  (mask * log_prob).sum(1)
            mean_log_prob_pos = pixel_sel_loc * mean_log_prob_pos
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = torch.div(loss.sum(),pixel_sel_loc.sum()+1e-4)
        else:

            mean_log_prob_pos = (mask * log_prob).sum(1)
            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.view(anchor_count, num_samples).mean()

        return loss

