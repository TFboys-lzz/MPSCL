import os
import sys
sys.path.append('..')
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from MPSCL.utils.func import dice_eval
from MPSCL.model.discriminator import get_discriminatord
from MPSCL.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from MPSCL.utils.func import loss_calc, bce_loss
from MPSCL.utils.loss import dice_loss,MPCL
from MPSCL.utils.func import prob_2_entropy,mpcl_loss_calc
import matplotlib.pyplot as plt
plt.switch_backend("agg")


def generate_pseudo_label(cla_feas_trg,class_centers,cfg):

    '''
    class_centers: C*N_fea
    cla_feas_trg: N*N_fea*H*W
    '''


    cla_feas_trg_de     = cla_feas_trg.detach()
    batch,N_fea,H,W     = cla_feas_trg_de.size()
    cla_feas_trg_de     = F.normalize(cla_feas_trg_de,p=2,dim=1)
    class_centers_norm  = F.normalize(class_centers,p=2,dim=1)
    cla_feas_trg_de     = cla_feas_trg_de.transpose(1,2).contiguous().transpose(2,3).contiguous() # N*H*W*N_fea
    cla_feas_trg_de     = torch.reshape(cla_feas_trg_de,[-1,N_fea])
    class_centers_norm  = class_centers_norm.transpose(0,1)  # N_fea*C
    batch_pixel_cosine  = torch.matmul(cla_feas_trg_de,class_centers_norm) #N*N_class
    threshold = cfg.TRAIN.PIXEL_SEL_TH
    pixel_mask          = pixel_selection(batch_pixel_cosine,threshold)
    hard_pixel_label    = torch.argmax(batch_pixel_cosine,dim=1)

    return hard_pixel_label,pixel_mask

def pixel_selection(batch_pixel_cosine,th):
    one_tag = torch.ones([1]).float().cuda()
    zero_tag = torch.zeros([1]).float().cuda()

    batch_sort_cosine,_ = torch.sort(batch_pixel_cosine,dim=1)
    pixel_sub_cosine    = batch_sort_cosine[:,-1]-batch_sort_cosine[:,-2]
    pixel_mask          = torch.where(pixel_sub_cosine>th,one_tag,zero_tag)

    return pixel_mask

def iter_eval(model,images_sval,labels_sval,images_target,labels_target,device,cfg):
    model.eval()

    with torch.no_grad():
        NUMCLASS  = cfg.NUM_CLASSES
        interp    = nn.Upsample(size=(256, 256), mode='bilinear',align_corners=True)
        cla_feas_src,pred_src_aux, pred_src_main = model(images_sval.cuda())

        pred_src_main = interp(pred_src_main)
        _,sval_dice_arr,sval_class_number  = dice_eval(pred=pred_src_main,label=labels_sval.cuda(),n_class=NUMCLASS)
        sval_dice_arr    = np.hstack(sval_dice_arr)
        cla_feas_trg,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        pred_trg_main  = interp(pred_trg_main)
        _,trg_dice_arr,trg_class_number = dice_eval(pred=pred_trg_main,label=labels_target.cuda(),n_class=NUMCLASS)
        trg_dice_arr   = np.hstack(trg_dice_arr)

        print('Dice')
        print('######## Source Validation Set ##########')
        print('Each Class Number {}'.format(sval_class_number))
        print('Myo:{:.3f}'.format(sval_dice_arr[1]))
        print('LAC:{:.3f}'.format(sval_dice_arr[2]))
        print('LVC:{:.3f}'.format(sval_dice_arr[3]))
        print('AA:{:.3f}'.format(sval_dice_arr[4]))
        print('######## Source Validation Set ##########')

        print('######## Target Train Set ##########')
        print('Each Class Number {}'.format(trg_class_number))
        print('Myo:{:.3f}'.format(trg_dice_arr[1]))
        print('LAC:{:.3f}'.format(trg_dice_arr[2]))
        print('LVC:{:.3f}'.format(trg_dice_arr[3]))
        print('AA:{:.3f}'.format(trg_dice_arr[4]))
        print('######## Target Train Set ##########')

def label_downsample(labels,fea_h,fea_w):

    '''
    labels: N*H*W
    '''
    labels = labels.float().cuda()
    labels = F.interpolate(labels, size=fea_w, mode='nearest')
    labels = labels.permute(0, 2, 1).contiguous()
    labels = F.interpolate(labels, size=fea_h, mode='nearest')
    labels = labels.permute(0, 2, 1).contiguous()  # n*fea_h*fea_w
    labels = labels.int()
    return labels

def update_class_center_iter(cla_src_feas,batch_src_labels,class_center_feas,m):

    '''
    batch_src_feas  : n*c*h*w
    barch_src_labels: n*h*w
    '''
    batch_src_feas     = cla_src_feas.detach()
    batch_src_labels   = batch_src_labels.cuda()
    n,c,fea_h,fea_w    = batch_src_feas.size()
    batch_y_downsample = label_downsample(batch_src_labels, fea_h, fea_w)  # n*fea_h*fea_w
    batch_y_downsample = batch_y_downsample.unsqueeze(1)  # n*1*fea_h*fea_w
    batch_class_center_fea_list = []
    for i in range(5):
        fea_mask        = torch.eq(batch_y_downsample,i).float().cuda()  #n*1*fea_h*fea_w
        class_feas      = batch_src_feas * fea_mask  # n*c*fea_h*fea_w
        class_fea_sum   = torch.sum(class_feas, [0, 2, 3])  # c
        class_num       = torch.sum(fea_mask, [0, 1, 2, 3])
        if class_num == 0:
            batch_class_center_fea = class_center_feas[i,:].detach()
        else:
            batch_class_center_fea = class_fea_sum/class_num
        batch_class_center_fea = batch_class_center_fea.unsqueeze(0) # 1 * c
        batch_class_center_fea_list.append(batch_class_center_fea)
    batch_class_center_feas = torch.cat(batch_class_center_fea_list,dim=0) # n_class * c
    class_center_feas = m * class_center_feas + (1-m) * batch_class_center_feas

    return class_center_feas

def train_mpscl(model, strain_loader,sval_loader,trgtrain_loader,cfg):
    '''
        UDA training
        '''
    # create the model and start the training

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    device            = cfg.GPU_ID
    num_classes       = cfg.NUM_CLASSES
    viz_tensorboard   = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)

    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMENTATION
    model.train()
    model.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # feature-level

    d_aux  = None
    d_main = None
    if cfg.TRAIN.D_NORM == 'Batch_Norm':
        norm_layer = nn.BatchNorm2d
    else:
        norm_layer = nn.InstanceNorm2d

    d_aux  = get_discriminatord(cfg.TRAIN.D_TYPE, 5, norm_layer, init_type='normal')
    d_main = get_discriminatord(cfg.TRAIN.D_TYPE, 5, norm_layer, init_type='normal')

    load_checkpoint(d_aux, checkpoint=cfg.TRAIN.D_AUX_RESTORE)
    load_checkpoint(d_main, checkpoint=cfg.TRAIN.D_MAIN_RESTORE)

    d_aux.train()
    d_aux.cuda()
    d_main.train()
    d_main.cuda()

    print('finish model setup')

    # optimized

    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    optimizer_d_aux  = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # compute class center
    class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    class_center_feas = torch.from_numpy(class_center_feas).float().cuda()

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning
    source_label         = 0
    targte_label         = 1
    strain_loader_iter   = enumerate(strain_loader)
    trgtrain_loader_iter = enumerate(trgtrain_loader)
    sval_loader_iter     = enumerate(sval_loader)


    mpcl_loss_src = MPCL(device,num_class=num_classes, temperature=cfg.TRAIN.SRC_TEMP,
                                       base_temperature=cfg.TRAIN.SRC_BASE_TEMP, m=cfg.TRAIN.SRC_MARGIN)

    mpcl_loss_trg = MPCL(device, num_class=num_classes, temperature=cfg.TRAIN.TRG_TEMP,
                                       base_temperature=cfg.TRAIN.TRG_BASE_TEMP, m=cfg.TRAIN.TRG_MARGIN)

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):


        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()

        # adapt LE if needed

        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)


        # UDA training
        # First only train segmentation network based on source label
        # Fix D  Update G
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False

        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:

            strain_loader_iter       = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()

        images_source, labels_source, _, = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch = trgtrain_loader_iter.__next__()
        images_target, labels_target, _ = batch

        # set train mode for each net
        model.train()
        d_main.train()
        d_aux.train()
        cla_feas_src, pred_src_aux, pred_src_main = model(images_source.cuda())
        cla_feas_trg, pred_trg_aux, pred_trg_main = model(images_target.cuda())

        class_center_feas = update_class_center_iter(cla_feas_src, labels_source, class_center_feas,m=cfg.TRAIN.CLASS_CENTER_M)

        hard_pixel_label,pixel_mask = generate_pseudo_label(cla_feas_trg, class_center_feas, cfg)


        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux          = interp(pred_src_aux)
            loss_seg_src_aux      = loss_calc(pred_src_aux, labels_source, cfg,device)
            loss_dice_src_aux     = dice_loss(pred_src_aux, labels_source, device)

        else:
            loss_seg_src_aux      = 0
            loss_dice_src_aux     = 0

        pred_src_main          = interp(pred_src_main)
        loss_seg_src_main      = loss_calc(pred_src_main, labels_source,cfg, device)
        loss_dice_src_main     = dice_loss(pred_src_main, labels_source, device)
        mpcl_loss_tr           = mpcl_loss_calc(feas=cla_feas_src,labels=labels_source,
                                                            class_center_feas=class_center_feas,
                                                            loss_func=mpcl_loss_src,tag='source')

        mpcl_loss_tg = mpcl_loss_calc(feas=cla_feas_trg, labels=hard_pixel_label,
                                                 class_center_feas=class_center_feas,
                                                 loss_func=mpcl_loss_trg,
                                                 pixel_sel_loc=pixel_mask, tag='target')

        seg_loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX    * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN  * loss_dice_src_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX   * loss_dice_src_aux
                + cfg.TRAIN.LAMBDA_MPCL_SRC       * mpcl_loss_tr
                + cfg.TRAIN.LAMBDA_MPCL_TRG       * mpcl_loss_tg)

        # adversarial training to fool the discriminator

        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux     = interp(pred_trg_aux)
            d_out_aux        = d_aux(prob_2_entropy(F.softmax(pred_trg_aux, dim=1)))
            loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
        else:
            loss_adv_trg_aux = 0

        pred_trg_main     = interp(pred_trg_main)
        d_out_main        = d_main(prob_2_entropy(F.softmax(pred_trg_main, dim=1)))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        adv_gloss         = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main +
                             cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)

        loss = seg_loss + adv_gloss
        loss.backward()

        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True

        # First we train d with source metadata
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux, dim=1)))
            loss_d_aux = bce_loss(d_out_aux, source_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()

        pred_src_main = pred_src_main.detach()
        d_out_main    = d_main(prob_2_entropy(F.softmax(pred_src_main, dim=1)))
        loss_d_main   = bce_loss(d_out_main, source_label)
        loss_d_main   = loss_d_main / 2
        loss_d_main.backward()

        # second we train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux, dim=1)))
            loss_d_aux = bce_loss(d_out_aux, targte_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0

        pred_trg_main = pred_trg_main.detach()
        d_out_main  = d_main(prob_2_entropy(F.softmax(pred_trg_main, dim=1)))
        loss_d_main = bce_loss(d_out_main, targte_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()


        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_mpcl_tr': mpcl_loss_tr,
                          'loss_mpcl_tg': mpcl_loss_tg,
                          'loss_dice_src_aux': loss_dice_src_aux,
                          'loss_dice_src_main': loss_dice_src_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main}


        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            class_center_feas_save_dir = cfg.TRAIN.SNAPSHOT_DIR + '/feas'
            os.makedirs(class_center_feas_save_dir,exist_ok=True)
            class_center_feas_save_pth =  f'{class_center_feas_save_dir}/class_center_feas_model_{i_iter}.npy'
            torch.save(model.state_dict(), snapshot_dir /  f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(), snapshot_dir /  f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            class_center_feas_npy = class_center_feas.cpu().detach().numpy()
            np.save(class_center_feas_save_pth,class_center_feas_npy)


            try:
                _, batch = sval_loader_iter.__next__()
            except StopIteration:
                sval_loader_iter = enumerate(sval_loader)
                _, batch = sval_loader_iter.__next__()
            images_sval, labels_sval, _, = batch

            iter_eval(model, images_sval, labels_sval, images_target, labels_target, device, cfg)


        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

def train_advent(model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''
    # create the model and start the training

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    device            = cfg.GPU_ID
    num_classes       = cfg.NUM_CLASSES
    viz_tensorboard   = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)

    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    #SEGMENTATION
    model.train()
    model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    # DISCRIMINATOR NETWORK
    # feature-level
    if cfg.TRAIN.D_NORM == 'Batch_Norm':
        norm_layer = nn.BatchNorm2d
    else:
        norm_layer = nn.InstanceNorm2d

    d_aux  = get_discriminatord(cfg.TRAIN.D_TYPE, 5, norm_layer, init_type='normal')
    d_main = get_discriminatord(cfg.TRAIN.D_TYPE, 5, norm_layer, init_type='normal')
    d_aux.train()
    d_aux.cuda()

    #output level
    d_main.train()
    d_main.cuda()
    print('finish model setup')
    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # interpolate output segmaps

    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning

    source_label = 0
    targte_label = 1
    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN


    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):


        # set train mode for each net
        model.train()
        d_main.train()
        d_aux.train()
        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LE if needed
        adjust_learning_rate(optimizer,i_iter,cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux,i_iter,cfg)
        adjust_learning_rate_discriminator(optimizer_d_main,i_iter,cfg)

        #UDA training
        # First only train segmentation network based on source label
        # set discriminator require_grad = False
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source metadata
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch
        _,pred_src_aux, pred_src_main = model(images_source.cuda())
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source,cfg,device)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source,device)
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0
        pred_src_main     = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source,cfg,device)
        loss_dice_main    = dice_loss(pred_src_main,labels_source,device)
        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux)
        loss.backward()

        # adversarial training to fool the discriminator
        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target,labels_target,_ = batch
        _,pred_trg_aux, pred_trg_main    = model(images_target.cuda())
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux     = interp(pred_trg_aux)
            d_out_aux        = d_aux(prob_2_entropy(F.softmax(pred_trg_aux,dim=1)))
            loss_adv_trg_aux = bce_loss(d_out_aux,source_label)
        else:
            loss_adv_trg_aux = 0
        pred_trg_main     = interp(pred_trg_main)
        d_out_main        = d_main(prob_2_entropy(F.softmax(pred_trg_main,dim=1)))
        loss_adv_trg_main = bce_loss(d_out_main,source_label)
        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main+
                cfg.TRAIN.LAMBDA_ADV_AUX  * loss_adv_trg_aux)
        loss.backward()

        #Train discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True

        #First we train d with source metadata
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            d_out_aux    = d_aux(prob_2_entropy(F.softmax(pred_src_aux,dim=1)))
            loss_d_aux   = bce_loss(d_out_aux,source_label)
            loss_d_aux   = loss_d_aux / 2
            loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main    = d_main(prob_2_entropy(F.softmax(pred_src_main,dim=1)))
        loss_d_main   = bce_loss(d_out_main,source_label)
        loss_d_main   = loss_d_main / 2
        loss_d_main.backward()

        # second we train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux    = d_aux(prob_2_entropy(F.softmax(pred_trg_aux,dim=1)))
            loss_d_aux   = bce_loss(d_out_aux,targte_label)
            loss_d_aux   = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main  = pred_trg_main.detach()
        d_out_main     = d_main(prob_2_entropy(F.softmax(pred_trg_main,dim=1)))
        loss_d_main    = bce_loss(d_out_main,targte_label)
        loss_d_main    = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                          'loss_adv_trg_aux' :loss_adv_trg_aux,
                          'loss_adv_trg_main':loss_adv_trg_main,
                          'loss_d_aux'       :loss_d_aux,
                          'loss_d_main'      :loss_d_main}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(),  snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(),  snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            try:
                _, batch = sval_loader_iter.__next__()
            except StopIteration:
                sval_loader_iter = enumerate(sval_loader)
                _, batch = sval_loader_iter.__next__()
            images_sval, labels_sval,_ = batch

            iter_eval(model, images_sval, labels_sval, images_target, labels_target, device,cfg)

        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer,current_losses,i_iter)

def log_losses_tensorboard(writer,current_losses,i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value),i_iter)

def print_losses(current_losses,i_iter):
    list_strings = []
    for loss_name,loss_value in current_losses.items():
        list_strings.append(f'{loss_name}={to_numpy(loss_value):.3f}')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter={i_iter} {full_string}')

def to_numpy(tensor):
    if isinstance(tensor,(int,float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def train_domain_adaptation(model,strain_loader,trgtrain_loader,sval_loader,cfg):

    if cfg.TRAIN.DA_METHOD == 'MPSCL':
        train_mpscl(model, strain_loader, sval_loader,trgtrain_loader, cfg)
    if cfg.TRAIN.DA_METHOD == 'AdvEnt':
        train_advent(model, strain_loader, sval_loader,trgtrain_loader, cfg)

def load_checkpoint(model, checkpoint,):
    saved_state_dict = torch.load(checkpoint,map_location='cpu')
    model.load_state_dict(saved_state_dict)

