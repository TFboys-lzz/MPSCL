import os



def make_datalist(data_fd, data_list,data_gt_list):

    filename_all = os.listdir(data_fd)

    filename_all = [data_fd + '/' + img_name + '\n' for img_name in filename_all if img_name.endswith('.npy')]

    gt_filename_all = [img_name + '_gt' for img_name in filename_all]

    with open(data_list, 'w') as fp:
        fp.writelines(filename_all)

    with open(data_gt_list, 'w') as fp:
        fp.writelines(gt_filename_all)




if __name__ == '__main__':

    #Plz change the path follow your setting
    data_fd      = '<root_dir>/train_mr/'
    data_list    = '../data/datalist/train_mr.txt'
    data_gt_list = '../data/datalist/train_mr_gt.txt'
    make_datalist(data_fd, data_list,data_gt_list)

