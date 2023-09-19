#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk


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
    RandomCrop,
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


class CMDA2DSliceDataset(Dataset):
    def __len__(self):
        return len(self.sample_2d_slice_csv)

    def __init__(self,
                 sample_2d_slice_csv_path,
                 mode='train',
                 annotation_type='labeled',
                 modality='t1',
                 data_num=-1,
                 use_aug=False,
                 train_crop_size=(256, 256)
                 ):

        assert mode in ['train', 'val']
        assert modality in ['t1', 't2']
        assert annotation_type in ['labeled', 'unlabeled']

        self.mode = mode
        self.use_aug = use_aug
        self.modal = modality
        self.annotation_type = annotation_type
        self.train_crop_size = train_crop_size

        self.sample_2d_slice_csv = pd.read_csv(sample_2d_slice_csv_path)

        if data_num == -1:
            pass
        else:
            self.sample_2d_slice_csv = self.sample_2d_slice_csv.sample(data_num)

        print(">>" * 30, 'modality: %s, anno type: %s' % (self.modal, self.annotation_type), "read:",
              sample_2d_slice_csv_path, 'data num: ', len(self.sample_2d_slice_csv), ">>" * 30)

    def __getitem__(self, index):
        # patient_id,slice_idx,t2/t1_path,seg_path
        current_select_row = self.sample_2d_slice_csv.iloc[index]

        current_patient_id = current_select_row['patient_id']
        current_slice_idx = current_select_row['slice_idx']

        # mr2ct
        if self.modal == 't1':
            current_feature_path = current_select_row['mr_path']
        elif self.modal == 't2':
            current_feature_path = current_select_row['ct_path']
        else:
            assert False

        # read array
        current_feature_array = sitk.GetArrayFromImage(sitk.ReadImage(current_feature_path))

        if self.annotation_type == 'labeled':
            current_seg_path = current_select_row['seg_path']
            current_seg_gt_array = sitk.GetArrayFromImage(sitk.ReadImage(current_seg_path))
        elif self.annotation_type == 'unlabeled':
            current_seg_gt_array = np.zeros_like(current_feature_array)
        else:
            assert False

        assert current_feature_array.shape == current_seg_gt_array.shape

        #
        current_img_size = current_feature_array.shape

        # #
        if self.use_aug and self.mode == 'train':
            current_aug_feature_array, current_aug_seg_gt_array = self.train_augmentation(current_feature_array,
                                                                                          current_seg_gt_array)
        else:
            current_aug_feature_array, current_aug_seg_gt_array = current_feature_array, current_seg_gt_array



        if False:
            lbvs = np.unique(current_aug_seg_gt_array)
            print('itensity min:%s, max:%s'%(current_aug_feature_array.min(),current_aug_feature_array.max()),'labels', lbvs)

            plt.figure(figsize=(12, 5))
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['axes.unicode_minus'] = False
            plt.suptitle('modal:%s %s, patient_id:%s, size: %s' % (
            self.modal, self.annotation_type, current_patient_id, str(current_img_size)))

            plt.subplot(1, 4, 1).set_title("origin_feature")
            plt.imshow(current_feature_array, cmap=plt.cm.jet)
            plt.colorbar()

            plt.subplot(1, 4, 2).set_title("origin_label")
            plt.imshow(current_seg_gt_array, cmap=plt.cm.jet)
            plt.colorbar()

            plt.subplot(1, 4, 3).set_title("aug_feature")
            plt.imshow(current_aug_feature_array, cmap=plt.cm.jet)
            plt.colorbar()

            plt.subplot(1, 4, 4).set_title("aug_label")
            plt.imshow(current_aug_seg_gt_array, cmap=plt.cm.jet)
            plt.colorbar()

            plt.show()

        # input no one-hot, one-hot while calculating dice loss

        # [1,W,H]
        feature_input_np_array = np.array([current_aug_feature_array]).astype(np.float)
        # [W,H]
        seg_np_array = np.array(current_aug_seg_gt_array).astype(np.int8)

        # t2
        feature_tensor = torch.from_numpy(feature_input_np_array).float()
        # no one-hot
        label_tensor = torch.from_numpy(seg_np_array).long()

        return feature_tensor, label_tensor

    @classmethod
    def train_augmentation(cls, img, mask):
        aug = Compose([
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            # RandomRotate90(p=0.5),
            # ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        ])

        auged = aug(image=img, mask=mask)
        return auged['image'], auged['mask']


from tqdm import tqdm
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # t1
    labeled_t1_slices_csv_path = '../../csv/all_mr_prep_slices_data.csv'

    # t2
    unlabeled_t2_slices_csv_path = '../../csv/all_ct_prep_slices_data.csv'

    # unpaired T1 and T2 dataset
    # t1
    labeled_t1_slices_dataset = CMDA2DSliceDataset(
        sample_2d_slice_csv_path=labeled_t1_slices_csv_path,
        mode='val',
        annotation_type='labeled',
        modality='t1',
        data_num=1,
        use_aug=True)

    unlabeled_t2_slices_dataset = CMDA2DSliceDataset(
        sample_2d_slice_csv_path=unlabeled_t2_slices_csv_path,
        mode='val',
        annotation_type='unlabeled',
        modality='t2',
        data_num=1,
        use_aug=True)

    # dataset unpaired
    batch_size = 1

    labeled_t1_slices_loader = DataLoader(labeled_t1_slices_dataset, shuffle=True, batch_size=batch_size,
                                          num_workers=0)
    unlabeled_t2_slices_loader = DataLoader(unlabeled_t2_slices_dataset, shuffle=True, batch_size=batch_size,
                                            num_workers=0)

    labeled_t1_slices_loader_iter = enumerate(labeled_t1_slices_loader)
    unlabeled_t2_slices_loader_iter = enumerate(unlabeled_t2_slices_loader)


    # check
    for t1_idx, (labeled_t1_feature_tensor, labeled_t1_seg_gt_tensor)  in tqdm(enumerate(labeled_t1_slices_loader), total=len(labeled_t1_slices_loader)):
        img_size = (labeled_t1_feature_tensor.shape[2], labeled_t1_feature_tensor.shape[3])
        if img_size[0] != 256 or img_size[1] != 256:
            print('t1 error: %s'%(str(img_size)))
        print(labeled_t1_feature_tensor.shape,labeled_t1_feature_tensor.min(), labeled_t1_feature_tensor.max())

    for t2_idx, (unlabeled_t2_feature_tensor, unlabeled_t2_label_tensor) in tqdm(enumerate(unlabeled_t2_slices_loader), total=len(unlabeled_t2_slices_loader)):
        img_size = (unlabeled_t2_feature_tensor.shape[2], unlabeled_t2_feature_tensor.shape[3])
        if img_size[0] != 256 or img_size[1] != 256:
            print('t2 error: %s' % (str(img_size)))
        print(unlabeled_t2_feature_tensor.shape, unlabeled_t2_feature_tensor.min(), unlabeled_t2_feature_tensor.max())


    # n_iter_each_epoch = 5
    # for iter_idx in tqdm(range(n_iter_each_epoch)):
    #     # t1
    #     t1_idx, (labeled_t1_feature_tensor, labeled_t1_seg_gt_tensor) = next(labeled_t1_slices_loader_iter)
    #     print('labeled_t1_feature_tensor, labeled_t1_seg_gt_tensor', labeled_t1_feature_tensor.size(), labeled_t1_seg_gt_tensor.size())
    #
    #     # t2
    #     t2_idx, (unlabeled_t2_feature_tensor, _) = next(unlabeled_t2_slices_loader_iter)
    #
    #     print('unlabeled_t2_feature_tensor', unlabeled_t2_feature_tensor.size())
    #     #
    #     # print(torch.max(labeled_t1_feature_tensor), torch.min(labeled_t1_feature_tensor))
    #     #
    #     # print(torch.max(unlabeled_t2_feature_tensor), torch.min(unlabeled_t2_feature_tensor))