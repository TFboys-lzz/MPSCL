## Dependencies
This code requires the following
* Python 3.6
* Pytorch 1.3.0

## Configure Dataset
* Thanks to [SIFA](https://github.com/cchen-cc/SIFA) for sharing the pre-processed data. We have changed the tfrecords data to Numpy. 
Plz download the data from [data](https://drive.google.com/drive/folders/1UFqj18A4vuoknldoqAkg9tx7S6CUjxRL) and put it in ./data folder
* Plz run ./dataset/create_datalist.py to create the file containing training data path.
* Plz run ./dataset/create_test_datalist.py to create the file containing testing data path.

## Configure Pretrained Model
* Plz download the pretrained model from [pretrained_model](https://drive.google.com/drive/folders/1UFqj18A4vuoknldoqAkg9tx7S6CUjxRL) and put it in ./pretrained_model folder
The pretrained model file contains two folder:

**training** contains the initialized models of our MPSCL for generating representative category prototypes and informative pseudo-labels, as described in the implementation details of our paper.
**testing**  contains the models corresponding to the results in our paper


## Training

To train MPSCL

* cd <root_dir>/MPSCL/scripts/

For MR2CT
* CUDA_VISIBLE_DEVICES=#device_id# python train.py --cfg ./configs/MPSCL_MR2CT.yml

For CT2MR
* CUDA_VISIBLE_DEVICES=#device_id# python train.py --cfg ./configs/MPSCL_CT2MR.yml

## Testing

To test MPSCL

**If you want to test our released pretrained model**

* cd <root_dir>/MPSCL/scripts

For MR2CT
* CUDA_VISIBLE_DEVICES=#device_id# python test.py --target_modality 'CT' --pretrained_model_pth '../pretrained_model/testing/MPSCL_MR2CT_best.pth'

For CT2MR
* CUDA_VISIBLE_DEVICES=#device_id# python test.py --target_modality 'MR' --pretrained_model_pth '../pretrained_model/testing/MPSCL_CT2MR_best.pth'

**If you want to test your model**

For MR2CT
* CUDA_VISIBLE_DEVICES=#device_id# python test.py --target_modality 'CT' --pretrained_model_pth 'your model path'

For CT2MR
* CUDA_VISIBLE_DEVICES=#device_id# python test.py --target_modality 'MR' --pretrained_model_pth 'your model path'


## Acknowledgements
This codebase is heavily borrowed from [AdvEnt](https://github.com/valeoai/ADVENT) and [SupContrast](https://github.com/HobbitLong/SupContrast)


