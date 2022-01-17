import numpy as np
import medpy.io as medio


def nii2npz():
    data_nii_pth  = '/metadata/zfzhu/liuzz/Python_Workspace/Medical_Image_Dataset/miccai_heart_segmentation/PnpAda_release_data/data_np/test_mr_image_labels/image_mr_1018.nii.gz'
    label_nii_pth = '/metadata/zfzhu/liuzz/Python_Workspace/Medical_Image_Dataset/miccai_heart_segmentation/PnpAda_release_data/data_np/test_mr_image_labels/gth_mr_1018.nii.gz'
    npz_pth       = '/metadata/zfzhu/liuzz/Python_Workspace/Medical_Image_Dataset/miccai_heart_segmentation/PnpAda_release_data/data_np/test_mr/image_mr_1018.nii.gz'

    data_arr, _  = medio.load(data_nii_pth)
    label_arr, _ = medio.load(label_nii_pth)

    np.savez(npz_pth, data_arr, label_arr)


if __name__=="__main__":


    nii2npz()