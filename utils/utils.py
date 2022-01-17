import numpy as np
import torch
import torch.nn as nn
def create_visual_anno(anno):
    """"""
    assert np.max(anno) <= 7, "only 7 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [0, 0, 0],
        1: [255, 248, 220],  # cornsilk
        2: [100, 149, 237],  # cornflowerblue
        3: [102, 205, 170],  # mediumAquamarine
        4: [205, 133, 63],  # peru
        5: [160, 32, 240],  # purple
        6: [255, 64, 64],  # brown1
        7: [139, 69, 19],  # Chocolate4
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno


def _compute_entropy_map(pred):

    '''
    pred: n*c*h*w
    '''
    n,c,h,w = pred.shape
    pred = torch.softmax(pred,dim=1)
    self_information_map =  -torch.mul(pred, torch.log2(pred + 1e-30)) / np.log2(c)
    entropy_map = torch.sum(self_information_map,dim=1) # n*h*w

    return entropy_map


