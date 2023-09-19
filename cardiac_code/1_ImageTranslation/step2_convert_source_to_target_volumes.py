#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import os
import pandas as pd
import numpy as np
from lib.utils.simple_parser import Parser

from lib.dataloader.medical_image_process import braincmda2022_cyclegan_normalization, braincmda2022_minmax_normalization, braincmda2022_zeroone_normalization

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.backends.cudnn as cudnn

import nibabel
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

import pickle

import SimpleITK as sitk


import copy

if __name__ == '__main__':
    yaml_config = Parser('./config/data_config/data_settings_splitting.yaml')

    gpu_list = [0]

    cut_yaml_config = Parser('./work_dir/ACMIT_SynModel/ACMIT_SynModel.yaml')
    cut_G_AB_weights = './work_dir/ACMIT_SynModel/model_weight/latest_net_G.pth'

    torch.cuda.set_device('cuda:{}'.format(gpu_list[0]))

    np.random.seed(cut_yaml_config['seed'])
    random.seed(cut_yaml_config['seed'])
    torch.cuda.manual_seed_all(cut_yaml_config['seed'])
    torch.random.manual_seed(cut_yaml_config['seed'])
    random.seed(cut_yaml_config['seed'])

    from lib.models.ACMIT_model import ACMIT

    G_AB = ACMIT(cut_yaml_config).netG

    print('load t1TOt2 generator: %s' % (cut_G_AB_weights))
    G_AB.load_state_dict(torch.load(cut_G_AB_weights, map_location='cpu'))
    # Fix G_AB
    G_AB = G_AB.cuda()
    G_AB.eval()
    for param in G_AB.parameters():
        param.requires_grad = False

    # Reading patient data
    # direction: 'mr2ct'
    all_brain_cmda_source_training_csv = pd.read_csv(yaml_config['all_mr_prep_volume_data_csv_path'])


    prep_fake_t2_data_dir = yaml_config['prep_fake_t2_data_dir']
    if not os.path.exists(prep_fake_t2_data_dir): os.makedirs(prep_fake_t2_data_dir)

    # 3D
    all_fake_t2_prep_volume_csv_path = yaml_config['all_fake_t2_prep_volume_csv_path']
    # 2D
    all_fake_t2_prep_slices_csv_path = yaml_config['all_fake_t2_prep_slices_csv_path']

    # fake_t2
    fake_t2_prep_3D_volume_list = []
    fake_t2_prep_2D_slice_list = []

    for idx, current_source_training_row in tqdm(all_brain_cmda_source_training_csv.iterrows(),
                                                 total=len(all_brain_cmda_source_training_csv),
                                                 desc='source_training prep'):
        # if idx > 1:
        #     break
        current_source_training_patient_id = current_source_training_row['patient_id']

        # mr2ct
        current_source_training_path = current_source_training_row['mr_path']
        current_seg_path = current_source_training_row['seg_path']

        target_prep_source_training_patient_id_dir = os.path.join(prep_fake_t2_data_dir, str(current_source_training_patient_id))

        if not os.path.exists(target_prep_source_training_patient_id_dir): os.makedirs(
            target_prep_source_training_patient_id_dir)

        img_array = sitk.GetArrayFromImage(sitk.ReadImage(current_source_training_path))
        seg_array = sitk.GetArrayFromImage(sitk.ReadImage(current_seg_path))

        origin_size = img_array.shape
        assert img_array.shape == seg_array.shape


        centre_crop_img_array = img_array
        centre_crop_seg_array = seg_array

        # # # # # # # # # # # # # # # #
        # fake t2 generation
        # # # # # # # # # # # # # # # #
        G_AB.eval()
        with torch.no_grad():
            current_real_t1_norm_volume = centre_crop_img_array

            # (155, 256, 256)
            current_fake_t2_norm_volume = np.zeros_like(current_real_t1_norm_volume)

            # batch_size

            current_patient_pred_slice_list = []
            slice_num = current_real_t1_norm_volume.shape[0]


            for current_slice_idx in tqdm(range(slice_num), total=slice_num, ncols=50,desc='pred'):
                # real_t1
                current_real_t1_map = current_real_t1_norm_volume[current_slice_idx]

                # # # # # # # # # # # # # #
                # # centre window start
                # # # # # # # # # # # # # #
                # [1,1,256,256]
                feature_np_array = np.array([[current_real_t1_map]])
                real_t1_feature_tensor = torch.from_numpy(feature_np_array).float()
                real_t1_feature_inputs = real_t1_feature_tensor.cuda(non_blocking=True)

                # predict
                # [1,1,256,256]
                fake_t2_tensor = G_AB(real_t1_feature_inputs)

                pred = fake_t2_tensor.cpu().numpy()
                del real_t1_feature_inputs

                centre_fake_t2_img = pred[0][0]

                # # # # # # # # # # # # # #

                current_fake_t2_norm_volume[current_slice_idx] = centre_fake_t2_img

            current_fake_t2_norm_volume = np.array(current_fake_t2_norm_volume)
            # print('current_fake_t2_norm_volume',current_fake_t2_norm_volume.shape,current_fake_t2_norm_volume.min(),current_fake_t2_norm_volume.max())
            # assert False
            # current_fake_t2_norm_volume = (current_fake_t2_norm_volume - current_fake_t2_norm_volume.min()) / (current_fake_t2_norm_volume.max() - current_fake_t2_norm_volume.min())
            assert current_fake_t2_norm_volume.shape == current_real_t1_norm_volume.shape


        centre_crop_fake_t2_array = current_fake_t2_norm_volume

        # current_img_resampling_array = centre_crop_img_array
        current_img_resampling_array = centre_crop_fake_t2_array

        current_seg_resampling_array = centre_crop_seg_array

        assert current_img_resampling_array.shape == current_seg_resampling_array.shape == (origin_size[0], 256, 256)


        save_prep_source_training_patient_id_img_nii_path = os.path.abspath(
            os.path.join(target_prep_source_training_patient_id_dir, '%s_prep_vol_img.nii.gz' % (current_source_training_patient_id)))
        save_prep_source_training_patient_id_seg_nii_path = os.path.abspath(
            os.path.join(target_prep_source_training_patient_id_dir, '%s_prep_vol_seg.nii.gz' % (current_source_training_patient_id)))

        res_img_o = sitk.GetImageFromArray(current_img_resampling_array)
        res_lb_o = sitk.GetImageFromArray(current_seg_resampling_array)
        sitk.WriteImage(res_img_o, save_prep_source_training_patient_id_img_nii_path)
        sitk.WriteImage(res_lb_o, save_prep_source_training_patient_id_seg_nii_path)

        current_target_size = current_img_resampling_array.shape


        fake_t2_prep_3D_volume_list.append(
            [current_source_training_patient_id, save_prep_source_training_patient_id_img_nii_path,
             save_prep_source_training_patient_id_seg_nii_path])

        # 2D slices
        target_prep_patient_id_2D_slices_dir = os.path.abspath(
            os.path.join(target_prep_source_training_patient_id_dir, 'slices'))

        if not os.path.exists(target_prep_patient_id_2D_slices_dir): os.makedirs(
            target_prep_patient_id_2D_slices_dir)

        for current_slice in range(current_img_resampling_array.shape[0]):
            current_slice_img_array = current_img_resampling_array[current_slice]
            current_slice_seg_array = current_seg_resampling_array[current_slice]

            if len(np.unique(current_slice_seg_array)) > 1:
                current_slice_contain_label = 1
            else:
                current_slice_contain_label = 0

            current_slice_img_itk = sitk.GetImageFromArray(current_slice_img_array)
            current_slice_seg_itk = sitk.GetImageFromArray(current_slice_seg_array)

            current_slice_img_nii_path = os.path.abspath(
                os.path.join(target_prep_patient_id_2D_slices_dir, 'slice_%s_img.nii.gz' % (current_slice)))
            current_slice_seg_nii_path = os.path.abspath(
                os.path.join(target_prep_patient_id_2D_slices_dir, 'slice_%s_seg.nii.gz' % (current_slice)))

            sitk.WriteImage(current_slice_img_itk, current_slice_img_nii_path)
            sitk.WriteImage(current_slice_seg_itk, current_slice_seg_nii_path)

            fake_t2_prep_2D_slice_list.append([current_source_training_patient_id, current_slice, current_slice_contain_label, current_slice_img_nii_path, current_slice_seg_nii_path])

    all_fake_t2_prep_volume_csv = pd.DataFrame(columns=["patient_id", "fake_t2_path", "seg_path"], data=fake_t2_prep_3D_volume_list)
    all_fake_t2_prep_volume_csv.to_csv(all_fake_t2_prep_volume_csv_path, index=False)

    all_fake_t2_prep_slices_csv = pd.DataFrame(columns=["patient_id", 'slice_idx', 'containLabel', "fake_t2_path", "seg_path"], data=fake_t2_prep_2D_slice_list)
    all_fake_t2_prep_slices_csv.to_csv(all_fake_t2_prep_slices_csv_path, index=False)


    #######################


