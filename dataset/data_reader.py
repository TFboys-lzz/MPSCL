from torch.utils.data import Dataset
import numpy as np

class CTDataset(Dataset):
    def __init__(self,data_pth,gt_pth,img_mean,transform=None,PL_Tag=False):

        with open(data_pth, 'r') as fp:
            self.ct_image_list = fp.readlines()

        if PL_Tag:
            self.ct_gt_list = None
        else:
            with open(gt_pth, 'r') as fp:
                self.ct_gt_list = fp.readlines()
        self.transform      = transform
        self.img_mean       = img_mean
        self.gen_pl         = PL_Tag   # generate pseudo label or not


    def __getitem__(self, index):

        if self.gen_pl:
            img_pth = self.ct_image_list[index][:-1]
            img     = self.gl_load_data(img_pth)
            gt      = np.zeros([img.shape[1],img.shape[2]],dtype=int)
        else:
            img_pth = self.ct_image_list[index][:-1]
            gt_pth  = self.ct_gt_list[index][:-1]
            img,gt  = self.load_data(img_pth,gt_pth)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.transpose(img, (2, 0, 1))  # 3*h*w

        gt = gt.astype(int)

        return img, gt,index

    def __len__(self):
        return len(self.ct_image_list)
    def load_data(self,img_pth, gt_pth):
        img = np.load(img_pth) # h*w*1
        gt  = np.load(gt_pth)  # h*w

        img = np.expand_dims(img,-1)
        img = np.tile(img,[1,1,3])  #h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img, gt

    def gl_load_data(self,img_pth):
        img = np.load(img_pth)  # h*w*1
        img = np.expand_dims(img,-1)
        img = np.tile(img, [1, 1, 3])  # h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img


class MRDataset(Dataset):
    def __init__(self, data_pth, gt_pth,img_mean, transform=None,PL_Tag=False):
        with open(data_pth, 'r') as fp:
            self.mr_image_list = fp.readlines()
        if PL_Tag:
            self.mr_gt_list = None

        else:
            with open(gt_pth, 'r') as fp:
                self.mr_gt_list = fp.readlines()
        self.transform      = transform
        self.img_mean       = img_mean
        self.gen_pl         = PL_Tag

    def __getitem__(self, index):
        if self.gen_pl:
            img_pth = self.mr_image_list[index][:-1]
            img = self.gl_load_data(img_pth)
            gt = np.zeros([img.shape[1], img.shape[2]], dtype=int)
        else:
            img_pth = self.mr_image_list[index][:-1]
            gt_pth = self.mr_gt_list[index][:-1]
            img, gt = self.load_data(img_pth, gt_pth)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.transpose(img, (2, 0, 1))  # 3*h*w

        gt = gt.astype(int)

        return img, gt, index

    def __len__(self):
        return len(self.mr_image_list)

    def load_data(self, img_pth, gt_pth):
        img = np.load(img_pth) # h*w*1
        gt  = np.load(gt_pth)
        img = np.expand_dims(img,-1)
        img = np.tile(img,[1,1,3]) # h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img, gt

    def gl_load_data(self,img_pth):
        img = np.load(img_pth)  # h*w*1
        img = np.expand_dims(img,-1)
        img = np.tile(img, [1, 1, 3])  # h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img

