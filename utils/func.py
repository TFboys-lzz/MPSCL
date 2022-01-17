import numpy as np
import torch
import torch.nn as nn
from MPSCL.utils.loss import cross_entropy_2d
# import cv2
import torch.nn.functional as F
import torch.sparse as sparse

def dice_eval(pred,label,n_class):
    '''
    pred:  b*c*h*w
    label: b*h*w
    '''
    pred     = torch.argmax(pred,dim=1)  # b*h*w
    dice     = 0
    dice_arr = []
    each_class_number = []
    eps      = 1e-7
    for i in range(n_class):
        A = (pred  == i)
        B = (label == i)
        each_class_number.append(torch.sum(B).cpu().data.numpy())
        inse  = torch.sum(A*B).float()
        union = (torch.sum(A)+torch.sum(B)).float()
        dice  += 2*inse/(union+eps)
        dice_arr.append((2*inse/(union+eps)).cpu().data.numpy())

    return dice,dice_arr,np.hstack(each_class_number)

def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

def loss_calc(pred,label,cfg,device):

    '''
    This function returns cross entropy loss for semantic segmentation
    '''
    # pred shape is batch * c * h * w
    # label shape is b*h*w
    label = label.long().cuda()
    return cross_entropy_2d(pred, label,cfg,device)


def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)

def _adjust_learning_rate(optimizer, i_iter, cfg, learning_rate):
    lr = lr_poly(learning_rate, i_iter, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate(optimizer, i_iter, cfg):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE)

def adjust_learning_rate_discriminator(optimizer, i_iter, cfg):
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE_D)

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-7)) / np.log2(c)

def sel_prob_2_entropy(prob):
    n, c, h, w = prob.size()
    weighted_self_info = -torch.mul(prob, torch.log2(prob + 1e-30)) / c
    entropy            = torch.sum(weighted_self_info,dim=1) #N*C*H*W
    # mean_entropy       = torch.sum(entropy,dim=[1,2])
    return entropy



def mpcl_loss_calc(feas,labels,class_center_feas,loss_func,
                               pixel_sel_loc=None,tag='source'):

    '''
    feas:  batch*c*h*w
    label: batch*img_h*img_w
    class_center_feas: n_class*n_feas
    '''

    n,c,fea_h,fea_w = feas.size()
    if tag == 'source':
        labels      = labels.float()
        labels      = F.interpolate(labels, size=fea_w, mode='nearest')
        labels      = labels.permute(0,2,1).contiguous()
        labels      = F.interpolate(labels, size=fea_h, mode='nearest')
        labels      = labels.permute(0, 2, 1).contiguous()         # batch*fea_h*fea_w

    labels  = labels.cuda()
    labels  = labels.view(-1).long()

    feas = torch.nn.functional.normalize(feas,p=2,dim=1)
    feas = feas.transpose(1,2).transpose(2,3).contiguous() #batch*c*h*w->batch*h*c*w->batch*h*w*c
    feas = torch.reshape(feas,[n*fea_h*fea_w,c]) # [batch*h*w] * c
    feas = feas.unsqueeze(1) # [batch*h*w] 1 * c

    class_center_feas = torch.nn.functional.normalize(class_center_feas,p=2,dim=1)
    class_center_feas = torch.transpose(class_center_feas, 0, 1)  # n_fea*n_class

    loss =  loss_func(feas,labels,class_center_feas,
                                                    pixel_sel_loc=pixel_sel_loc)
    return loss


