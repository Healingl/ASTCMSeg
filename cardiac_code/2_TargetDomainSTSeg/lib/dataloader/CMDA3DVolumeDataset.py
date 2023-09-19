#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms
import numpy as np
import random
from lib.dataloader.medical_loader_utils import get_sample_area_by_centre, get_around_mask_order_sample_list
from lib.dataloader.medical_image_process import crop_cube_from_volume

from lib.dataloader.medical_image_process import braincmda2022_onezero_to_meanstd_normalization, braincmda2022_meanstd_to_onezero_normalization

from lib.utils.data_process import get_ND_bounding_box
from sklearn.utils import shuffle
import lib.augment as augment3D
import cv2
from matplotlib import pyplot as plt
import SimpleITK as sitk

import volumentations as V


from albumentations import (
    Compose,
    OneOf,
    Flip,
    PadIfNeeded,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    MotionBlur,
    OpticalDistortion,
    RandomSizedCrop,
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    ShiftScaleRotate,
    CenterCrop,
    Transpose,
    GridDistortion,
    ElasticTransform,
    RandomGamma,
    RandomBrightnessContrast,
    RandomContrast,
    RandomBrightness,
    CLAHE,
    HueSaturationValue,
    Blur,
    MedianBlur,
    ChannelShuffle,
)


def padding_array_by_3D_cropping_window(input_array, crop_size, is_sample=False, constant_values=-1):
    origin_z_size, origin_y_size, origin_x_size = input_array.shape
    crop_z_size, crop_y_size, crop_x_size = crop_size

    # small origin size
    if origin_z_size < crop_z_size or origin_y_size < crop_y_size or origin_x_size < crop_x_size:

        lack_z_pad_num = crop_z_size - origin_z_size
        lack_pz_left = max(lack_z_pad_num // 2, 0)
        lack_pz_right = max(lack_z_pad_num, 0) - lack_pz_left

        lack_y_pad_num = crop_y_size - origin_y_size
        lack_py_left = max(lack_y_pad_num // 2, 0)
        lack_py_right = max(lack_y_pad_num, 0) - lack_py_left

        lack_x_pad_num = crop_x_size - origin_x_size
        lack_px_left = max(lack_x_pad_num // 2, 0)
        lack_px_right = max(lack_x_pad_num, 0) - lack_px_left

        input_array = np.pad(input_array, [(lack_pz_left, lack_pz_right), (lack_py_left, lack_py_right),
                                           (lack_px_left, lack_px_right)],
                             mode='constant', constant_values=constant_values)
    else:
        lack_pz_left, lack_pz_right, lack_py_left, lack_py_right, lack_px_left, lack_px_right = 0, 0, 0, 0, 0, 0

    if is_sample:
        return input_array, [(lack_pz_left, lack_pz_right), (lack_py_left, lack_py_right),
                             (lack_px_left, lack_px_right)]

    origin_z_size, origin_y_size, origin_x_size = input_array.shape

    if origin_z_size % crop_z_size != 0:
        z_pad_num = crop_z_size - origin_z_size % crop_z_size
        pz_left = max(z_pad_num // 2, 0)
        pz_right = max(z_pad_num, 0) - pz_left
    else:
        pz_left = 0
        pz_right = 0

    if origin_y_size % crop_y_size != 0:
        y_pad_num = crop_y_size - origin_y_size % crop_y_size
        py_left = max(y_pad_num // 2, 0)
        py_right = max(y_pad_num, 0) - py_left
    else:
        py_left = 0
        py_right = 0

    if origin_x_size % crop_x_size != 0:
        x_pad_num = crop_x_size - origin_x_size % crop_x_size
        px_left = max(x_pad_num // 2, 0)
        px_right = max(x_pad_num, 0) - px_left
    else:
        px_left = 0
        px_right = 0

    padding_array = np.pad(input_array, [(pz_left, pz_right), (py_left, py_right), (px_left, px_right)],
                           mode='constant', constant_values=constant_values)

    z_left_padding_len = lack_pz_left + pz_left
    z_right_padding_len = lack_pz_right + pz_right

    y_left_padding_len = lack_py_left + py_left
    y_right_padding_len = lack_py_right + py_right

    x_left_padding_len = lack_px_left + px_left
    x_right_padding_len = lack_px_right + px_right

    return padding_array, [(z_left_padding_len, z_right_padding_len), (y_left_padding_len, y_right_padding_len),
                           (x_left_padding_len, x_right_padding_len)]


def reverse_padding_array_to_3D_origin_array(padding_array, padding_list):
    assert len(padding_array.shape) == len(padding_list)
    padding_img_size_z, padding_img_size_y, padding_img_size_x = padding_array.shape

    return padding_array[padding_list[0][0]:padding_img_size_z - padding_list[0][1],
           padding_list[1][0]:padding_img_size_y - padding_list[1][1],
           padding_list[2][0]:padding_img_size_x - padding_list[2][1]]


class CMDA3DVolumeDataset(Dataset):
    def __len__(self):
        return len(self.sample_3d_volume_csv)

    def __init__(self,
                 sample_3d_volume_csv_path,
                 mode='train',
                 annotation_type = 'labeled',
                 modality='fake_t2',
                 data_num=-1,
                 use_aug=False,
                 crop_size=(16, 256, 256),
                 step_size=(8, 8, 8),
                 crop_type='mask_random',
                 sup=False,
                 prep_target_plabel_data_dir = 'None',
                 ):

        assert mode in ['train', 'val']
        assert modality in ['fake_t2', 't1', 't2']
        assert annotation_type in ['labeled','unlabeled']
        assert crop_type in ['random', 'mask_random' , 'centre']

        self.mode = mode
        self.use_aug = use_aug
        self.modal = modality
        self.annotation_type = annotation_type
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.step_size = step_size
        self.sup=sup

        assert len(self.crop_size) == 3

        self.sample_3d_volume_csv = pd.read_csv(sample_3d_volume_csv_path)

        # self.prep_target_plabel_data_dir = '/mnt/data4/zyz/UDA3DSeg/prep_cardiac_data/VAST_mr2ct_target_label/'
        self.prep_target_plabel_data_dir = prep_target_plabel_data_dir

        if os.path.isdir(self.prep_target_plabel_data_dir):
            print('load pseudo labels dir:', self.prep_target_plabel_data_dir)
        else:
            print('no pseudo labels only sup training!')


        if self.use_aug and mode == 'train':

            self.aug = V.Compose([
                V.ElasticTransform((0, 0.25), interpolation=1, p=0.1),
                V.Flip(0, p=0.5),
                V.Flip(1, p=0.5),
                V.Flip(2, p=0.5),
                V.RandomRotate90((1, 2), p=0.5),

            ])

        if data_num == -1:
            pass
        else:
            self.sample_3d_volume_csv = self.sample_3d_volume_csv.sample(data_num)

        print(">>" * 30, 'modality: %s, anno type: %s'%(self.modal, self.annotation_type), "read:", sample_3d_volume_csv_path, 'data num: ',len(self.sample_3d_volume_csv), ">>" * 30)

    def RandomSaturation(self, img, saturation_limit=[0.9,1.1]):
        saturation=random.uniform(saturation_limit[0], saturation_limit[1])
        return np.clip(img*saturation,0,1)

    def RandomBrightness(self, img, intensity_limit=[0, 0.1]):
        brightness=random.uniform(intensity_limit[0], intensity_limit[1])
        return np.clip(img+brightness,0,1)

    def RandomContrast(self,img,contrast_limit=[0.9,1.1]):
        mean=np.mean(img,axis=(0,1,2),keepdims=True)
        contrast = random.uniform(contrast_limit[0], contrast_limit[1])
        return np.clip(img * contrast + mean * (1 - contrast),0,1)

    def ColorJetter(self,img,p=0.5):
        if random.random()>p:
            augs=[self.RandomSaturation,self.RandomBrightness,self.RandomContrast]
            random.shuffle(augs)
            for aug in augs:
                if random.random()>0.5:
                    img=aug(img)
            return img
        else:
            return img
    def __getitem__(self, index):

        # patient_id,slice_idx,,seg_path
        current_select_row = self.sample_3d_volume_csv.iloc[index]

        current_patient_id = current_select_row['patient_id']

        # ct2mr
        if self.modal == 'fake_t2':
            current_feature_path = current_select_row['fake_t2_path']

        elif self.modal == 't1':
            current_feature_path = current_select_row['mr_path']
        elif self.modal == 't2':
            current_feature_path = current_select_row['ct_path']
        else:
            assert False

        # read array
        # [-1, 1]
        current_feature_array = sitk.GetArrayFromImage(sitk.ReadImage(current_feature_path))

        # convert to [0, 1]
        current_feature_array = braincmda2022_meanstd_to_onezero_normalization(current_feature_array)

        if self.sup:
            current_seg_path = current_select_row['seg_path']
            current_seg_gt_array = sitk.GetArrayFromImage(sitk.ReadImage(current_seg_path))
        else:
            if self.annotation_type == 'labeled':
                if self.modal == 't1' or self.modal == 'fake_t2':
                    current_seg_path = current_select_row['seg_path']
                    current_seg_gt_array = sitk.GetArrayFromImage(sitk.ReadImage(current_seg_path))
                else:
                    target_training_plabel_dir = os.path.join(self.prep_target_plabel_data_dir, str(current_patient_id))
                    if os.path.exists(target_training_plabel_dir):
                        current_seg_path = os.path.abspath(os.path.join(target_training_plabel_dir, '%s_prep_vol_seg.nii.gz' % (current_patient_id)))
                        current_seg_gt_array = sitk.GetArrayFromImage(sitk.ReadImage(current_seg_path))
                    else:
                        thr = np.percentile(current_feature_array.ravel(), 90)
                        current_seg_gt_array = np.zeros_like(current_feature_array)
                        # current_seg_gt_array[current_feature_array>thr] = 1



            elif self.annotation_type == 'unlabeled':
                current_seg_gt_array = np.zeros_like(current_feature_array)
            else:
                assert False

        assert current_feature_array.shape == current_seg_gt_array.shape



        # no padding, limit size

        # step1: padding to 256x
        origin_size = current_feature_array.shape
        padding_feature_array, _ = padding_array_by_3D_cropping_window(current_feature_array,
                                                                       self.crop_size,
                                                                       is_sample=True,
                                                                       constant_values=np.min(current_feature_array))
        padding_seg_gt_array, _ = padding_array_by_3D_cropping_window(current_seg_gt_array,
                                                                      self.crop_size,
                                                                      is_sample=True,
                                                                      constant_values=0)

        padding_size = padding_feature_array.shape

        assert padding_size[0] >= self.crop_size[0], "current current_volume_size[0]: %s" % (
            padding_size[0])
        assert padding_size[1] >= self.crop_size[1], "current current_volume_size[1]: %s" % (
            padding_size[1])
        assert padding_size[2] >= self.crop_size[2], "current current_volume_size[2]: %s" % (
            padding_size[2])

        # padding_feature_array = current_feature_array
        # padding_seg_gt_array = current_seg_gt_array

        assert padding_feature_array.shape == padding_seg_gt_array.shape

        if False:
            lbvs = np.unique(padding_seg_gt_array)
            print('labels', lbvs)
            test_slice_idx = current_feature_array.shape[0] // 2

            plt.figure(figsize=(12, 5))
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['axes.unicode_minus'] = False
            plt.suptitle('modal:%s, origin size: %s, padding size: %s, slice:%s' % (self.modal,
                                                                                    str(origin_size),
                                                                                    str(padding_size),
                                                                                    test_slice_idx))

            plt.subplot(1, 4, 1).set_title("current_feature_array")
            plt.imshow(current_feature_array[test_slice_idx], cmap=plt.cm.jet)
            plt.colorbar()

            plt.subplot(1, 4, 2).set_title("current_seg_gt_array")
            plt.imshow(current_seg_gt_array[test_slice_idx], cmap=plt.cm.jet)
            plt.colorbar()

            plt.subplot(1, 4, 3).set_title("padding_feature_array")
            plt.imshow(padding_feature_array[test_slice_idx], cmap=plt.cm.jet)
            plt.colorbar()

            plt.subplot(1, 4, 4).set_title("padding_seg_gt_array")
            plt.imshow(padding_seg_gt_array[test_slice_idx], cmap=plt.cm.jet)
            plt.colorbar()

            plt.show()

        # step2: crop feature

        # random crop
        # 0~padding_feature_array.shape[0] - self.crop_size[0]
        if self.crop_type == 'random':

            crop_z_point = np.random.randint(0, padding_feature_array.shape[0] - self.crop_size[0] + 1)
            crop_y_point = np.random.randint(0, padding_feature_array.shape[1] - self.crop_size[1] + 1)
            crop_x_point = np.random.randint(0, padding_feature_array.shape[2] - self.crop_size[2] + 1)
            current_random_crop = (crop_z_point, crop_y_point, crop_x_point)



        else:
            assert False

        """
        current_sample_feature_array = padding_feature_array[crop_z_point:crop_z_point + self.crop_size[0],
                                       crop_y_point:crop_y_point + self.crop_size[1],
                                       crop_x_point:crop_x_point + self.crop_size[2]]
        current_sample_seg_gt_array = padding_seg_gt_array[crop_z_point:crop_z_point + self.crop_size[0],
                                      crop_y_point:crop_y_point + self.crop_size[1],
                                      crop_x_point:crop_x_point + self.crop_size[2]]
        """

        current_sample_feature_array = crop_cube_from_volume(origin_volume=padding_feature_array,
                                                             crop_point=current_random_crop,
                                                             crop_size=self.crop_size)

        current_sample_seg_gt_array = crop_cube_from_volume(origin_volume=padding_seg_gt_array,
                                                             crop_point=current_random_crop,
                                                             crop_size=self.crop_size)

        assert current_sample_feature_array.shape == current_sample_seg_gt_array.shape == (
        self.crop_size[0], self.crop_size[1], self.crop_size[2])

        # #
        if self.use_aug and self.mode == 'train':
            # [current_aug_feature_array], current_aug_seg_gt_array = self.transform([current_sample_feature_array],
            #                                                                        current_sample_seg_gt_array)

            # current_sample_feature_array = self.ColorJetter(current_sample_feature_array)
            seg_augmented = self.aug(image=current_sample_feature_array, mask=current_sample_seg_gt_array)
            current_aug_feature_array = seg_augmented['image']
            current_aug_feature_array = self.ColorJetter(current_aug_feature_array)
            current_aug_seg_gt_array = seg_augmented['mask']

            # current_aug_seg_gt_array[current_aug_seg_gt_array==255] = 0

        else:
            current_aug_feature_array, current_aug_seg_gt_array = current_sample_feature_array, current_sample_seg_gt_array


        # 0~1 to -1~1
        current_aug_feature_array = braincmda2022_onezero_to_meanstd_normalization(current_aug_feature_array)


        if False:

            lbvs = np.unique(current_aug_seg_gt_array)
            print('labels', lbvs)
            test_slice_idx = current_sample_feature_array.shape[0] // 2
            plt.figure(figsize=(12, 5))
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['axes.unicode_minus'] = False
            plt.suptitle('modal:%s %s, patient_id:%s, size: %s' % (
                self.modal, self.annotation_type, current_patient_id, str(current_sample_feature_array.shape)))

            plt.subplot(1, 4, 1).set_title("origin_feature")
            plt.imshow(current_sample_feature_array[test_slice_idx], cmap=plt.cm.jet)
            plt.colorbar()

            plt.subplot(1, 4, 2).set_title("origin_label")
            plt.imshow(current_sample_seg_gt_array[test_slice_idx], cmap=plt.cm.jet)
            plt.colorbar()

            plt.subplot(1, 4, 3).set_title("aug_feature")
            plt.imshow(current_aug_feature_array[test_slice_idx], cmap=plt.cm.jet)
            plt.colorbar()

            plt.subplot(1, 4, 4).set_title("aug_label")
            plt.imshow(current_aug_seg_gt_array[test_slice_idx], cmap=plt.cm.jet)
            plt.colorbar()

            plt.show()

        # input no one-hot, one-hot while calculating dice loss
        # [1,D,W,H]
        feature_input_np_array = np.array([current_aug_feature_array]).astype(np.float)
        # [D,W,H]
        seg_np_array = np.array(current_aug_seg_gt_array).astype(np.int)

        # t2
        feature_tensor = torch.from_numpy(feature_input_np_array).float()
        # no one-hot
        label_tensor = torch.from_numpy(seg_np_array).long()


        return feature_tensor, label_tensor



from tqdm import tqdm
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # fake t2
    labeled_fake_t2_volume_csv_path = '../../csv/all_fake_mr2ct_ACMIT_prep_volume.csv'

    # t2
    unlabeled_real_t2_volume_csv_path = '../../csv/all_ct_prep_volume_data.csv'

    tumor_crop_size = (32, 256, 256)
    step_size = (16, 16, 16)

    # unpaired fake_t2 and t2 dataset
    # fake_t2
    labeled_fake_t2_volume_dataset = CMDA3DVolumeDataset(
                                 sample_3d_volume_csv_path=labeled_fake_t2_volume_csv_path,
                                 mode='train',
                                 annotation_type = 'labeled',
                                 modality='fake_t2',
                                 data_num=-1,
                                 use_aug=True,
                                 crop_size=tumor_crop_size,
                                 step_size=step_size,
                                 crop_type='random')

    unlabeled_t2_volume_dataset = CMDA3DVolumeDataset(
                                sample_3d_volume_csv_path=unlabeled_real_t2_volume_csv_path,
                                mode='train',
                                annotation_type = 'labeled',
                                modality='t2',
                                data_num=-1,
                                use_aug=True,
                                crop_size=tumor_crop_size,
                                step_size=step_size,
                                crop_type='random')

    # dataset unpaired
    batch_size = 1

    labeled_fake_t2_volume_loader = DataLoader(labeled_fake_t2_volume_dataset,
                                               shuffle=True,
                                               batch_size=batch_size,
                                               num_workers=0)
    unlabeled_t2_volume_loader = DataLoader(unlabeled_t2_volume_dataset,
                                            shuffle=True,
                                            batch_size=batch_size,
                                            num_workers=0)


    labeled_fake_t2_volume_loader_iter = enumerate(labeled_fake_t2_volume_loader)
    unlabeled_t2_volume_loader_iter = enumerate(unlabeled_t2_volume_loader)
    n_iter_each_epoch = len(labeled_fake_t2_volume_loader)
    for iter_idx in tqdm(range(n_iter_each_epoch)):

        # fake_t2
        fake_t2_idx, (labeled_fake_t2_feature_tensor, labeled_fake_t2_seg_gt_tensor) = next(labeled_fake_t2_volume_loader_iter)
        print('labeled_fake_t2_feature_tensor, labeled_fake_t2_seg_gt_tensor', labeled_fake_t2_feature_tensor.size(), labeled_fake_t2_seg_gt_tensor.size())
        lbvs = np.unique(labeled_fake_t2_seg_gt_tensor.cpu().numpy())
        print('lbvs',lbvs)


        # # # # t2
        # t2_idx, (unlabeled_t2_feature_tensor, unlabeled_t2_seg_gt_tensor) = next(unlabeled_t2_volume_loader_iter)
        # #
        # print('unlabeled_t2_feature_tensor', unlabeled_t2_feature_tensor.size())
        # lbvs = np.unique(unlabeled_t2_seg_gt_tensor.cpu().numpy())
        # print('lbvs',lbvs)


