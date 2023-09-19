#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import os
import nibabel
import pandas as pd
import numpy as np
from copy import copy
import cv2
import random
from scipy import ndimage
import SimpleITK as sitk
import copy
from tqdm import tqdm
from matplotlib import pyplot as plt
import shutil
import nibabel as nib
from simple_parser import Parser



if __name__ == '__main__':
    yaml_config = Parser('./config/data_config/data_settings_splitting.yaml')

    prep_ct_data_dir = yaml_config['prep_ct_data_dir']
    prep_mr_data_dir = yaml_config['prep_mr_data_dir']

    if not os.path.exists(prep_ct_data_dir):os.makedirs(prep_ct_data_dir)
    if not os.path.exists(prep_mr_data_dir):os.makedirs(prep_mr_data_dir)

    all_ct_npz_data_csv = pd.read_csv(yaml_config['all_ct_npz_data_csv_path'])
    all_mr_npz_data_csv = pd.read_csv(yaml_config['all_mr_npz_data_csv_path'])

    all_ct_prep_volume_data_csv_path = yaml_config['all_ct_prep_volume_data_csv_path']
    all_mr_prep_volume_data_csv_path = yaml_config['all_mr_prep_volume_data_csv_path']

    all_mr_prep_slices_data_csv_path = yaml_config['all_mr_prep_slices_data_csv_path']
    all_ct_prep_slices_data_csv_path = yaml_config['all_ct_prep_slices_data_csv_path']

    # ct
    ct_prep_volume_data_list = []
    ct_prep_slices_data_list = []


    for idx, current_ct_row in tqdm(all_ct_npz_data_csv.iterrows(), total=len(all_ct_npz_data_csv), desc='ct_npz'):
        # if idx > 1:
        #     break
        current_ct_patient_id = current_ct_row['patient_id']
        current_ct_npz_path = current_ct_row['npz_file_path']

        target_prep_ct_patient_id_dir = os.path.join(prep_ct_data_dir, str(current_ct_patient_id))

        if not os.path.exists(target_prep_ct_patient_id_dir): os.makedirs(target_prep_ct_patient_id_dir)

        ## image and mask array
        current_ct_npz = np.load(current_ct_npz_path)
        current_ct_img_array = current_ct_npz['arr_0']
        current_ct_seg_array = current_ct_npz['arr_1']

        # min_max_norm and scale
        current_ct_normal_array = (current_ct_img_array - current_ct_img_array.min()) / (
                    current_ct_img_array.max() - current_ct_img_array.min())
        current_ct_normal_array = current_ct_normal_array*2 - 1


        assert current_ct_normal_array.shape == current_ct_seg_array.shape

        target_prep_ct_patient_id_img_nii_gz_path = os.path.abspath(
            os.path.join(target_prep_ct_patient_id_dir, '%s_ct_img.nii.gz' % (current_ct_patient_id)))
        target_prep_ct_patient_id_seg_nii_gz_path = os.path.abspath(
            os.path.join(target_prep_ct_patient_id_dir, '%s_ct_seg.nii.gz' % (current_ct_patient_id)))


        res_img_o = sitk.GetImageFromArray(current_ct_normal_array)
        res_lb_o = sitk.GetImageFromArray(current_ct_seg_array)
        sitk.WriteImage(res_img_o, target_prep_ct_patient_id_img_nii_gz_path)
        sitk.WriteImage(res_lb_o, target_prep_ct_patient_id_seg_nii_gz_path)

        lbvs = np.unique(current_ct_seg_array)
        print('labels',lbvs)

        ct_prep_volume_data_list.append([current_ct_patient_id,
                                  target_prep_ct_patient_id_img_nii_gz_path,
                                  target_prep_ct_patient_id_seg_nii_gz_path])

        # 2D slices
        target_prep_ct_patient_id_2D_slices_dir = os.path.abspath(
            os.path.join(target_prep_ct_patient_id_dir, 'slices'))
        if not os.path.exists(target_prep_ct_patient_id_2D_slices_dir): os.makedirs(
            target_prep_ct_patient_id_2D_slices_dir)
        for current_slice in range(current_ct_normal_array.shape[0]):
            current_slice_ct_array = current_ct_normal_array[current_slice]
            current_slice_seg_array = current_ct_seg_array[current_slice]

            current_slice_ct_nii_gz_path = os.path.abspath(
                os.path.join(target_prep_ct_patient_id_2D_slices_dir, 'slice_%s_ct.nii.gz' % (current_slice)))
            current_slice_seg_nii_gz_path = os.path.abspath(
                os.path.join(target_prep_ct_patient_id_2D_slices_dir, 'slice_%s_seg.nii.gz' % (current_slice)))

            slice_img_o = sitk.GetImageFromArray(current_slice_ct_array)
            slice_lb_o = sitk.GetImageFromArray(current_slice_seg_array)
            sitk.WriteImage(slice_img_o, current_slice_ct_nii_gz_path)
            sitk.WriteImage(slice_lb_o, current_slice_seg_nii_gz_path)

            ct_prep_slices_data_list.append(
                [current_ct_patient_id, current_slice, current_slice_ct_nii_gz_path, current_slice_seg_nii_gz_path])

    all_ct_prep_volume_data_csv = pd.DataFrame(columns=["patient_id", "ct_path", "seg_path"], data=ct_prep_volume_data_list)
    all_ct_prep_volume_data_csv.to_csv(all_ct_prep_volume_data_csv_path, index=False)

    all_ct_prep_slices_data_csv = pd.DataFrame(columns=["patient_id", 'slice_idx', "ct_path", "seg_path"],
                                               data=ct_prep_slices_data_list)
    all_ct_prep_slices_data_csv.to_csv(all_ct_prep_slices_data_csv_path, index=False)

    # mr
    mr_prep_volume_data_list = []
    mr_prep_slices_data_list = []

    for idx, current_mr_row in tqdm(all_mr_npz_data_csv.iterrows(), total=len(all_mr_npz_data_csv), desc='mr_npz'):
        # if idx > 1:
        #     break

        current_mr_patient_id = current_mr_row['patient_id']
        current_mr_npz_path = current_mr_row['npz_file_path']

        target_prep_mr_patient_id_dir = os.path.join(prep_mr_data_dir, str(current_mr_patient_id))

        if not os.path.exists(target_prep_mr_patient_id_dir): os.makedirs(target_prep_mr_patient_id_dir)

        ## image and mask array
        current_mr_npz = np.load(current_mr_npz_path)
        current_mr_img_array = current_mr_npz['arr_0']
        current_mr_seg_array = np.array(current_mr_npz['arr_1']).astype(np.int8)

        # min_max_norm and scale
        current_mr_normal_array = (current_mr_img_array - current_mr_img_array.min())/(current_mr_img_array.max()-current_mr_img_array.min())
        current_mr_normal_array = current_mr_normal_array*2 - 1

        assert current_mr_normal_array.shape == current_mr_seg_array.shape

        target_prep_mr_patient_id_img_nii_gz_path = os.path.abspath(
            os.path.join(target_prep_mr_patient_id_dir, '%s_mr_img.nii.gz' % (current_mr_patient_id)))
        target_prep_mr_patient_id_seg_nii_gz_path = os.path.abspath(
            os.path.join(target_prep_mr_patient_id_dir, '%s_mr_seg.nii.gz' % (current_mr_patient_id)))



        res_img_o = sitk.GetImageFromArray(current_mr_normal_array)
        res_lb_o = sitk.GetImageFromArray(current_mr_seg_array)
        sitk.WriteImage(res_img_o, target_prep_mr_patient_id_img_nii_gz_path)
        sitk.WriteImage(res_lb_o, target_prep_mr_patient_id_seg_nii_gz_path)

        lbvs = np.unique(current_mr_seg_array)
        print('labels', lbvs)

        mr_prep_volume_data_list.append([current_mr_patient_id,
                                         target_prep_mr_patient_id_img_nii_gz_path,
                                         target_prep_mr_patient_id_seg_nii_gz_path])

        # 2D slices
        target_prep_mr_patient_id_2D_slices_dir = os.path.abspath(
            os.path.join(target_prep_mr_patient_id_dir, 'slices'))
        if not os.path.exists(target_prep_mr_patient_id_2D_slices_dir): os.makedirs(
            target_prep_mr_patient_id_2D_slices_dir)
        for current_slice in range(current_mr_normal_array.shape[0]):
            current_slice_mr_array = current_mr_normal_array[current_slice]
            current_slice_seg_array = current_mr_seg_array[current_slice]

            current_slice_mr_nii_gz_path = os.path.abspath(
                os.path.join(target_prep_mr_patient_id_2D_slices_dir, 'slice_%s_mr.nii.gz' % (current_slice)))
            current_slice_seg_nii_gz_path = os.path.abspath(
                os.path.join(target_prep_mr_patient_id_2D_slices_dir, 'slice_%s_seg.nii.gz' % (current_slice)))

            slice_img_o = sitk.GetImageFromArray(current_slice_mr_array)
            slice_lb_o = sitk.GetImageFromArray(current_slice_seg_array)
            sitk.WriteImage(slice_img_o, current_slice_mr_nii_gz_path)
            sitk.WriteImage(slice_lb_o, current_slice_seg_nii_gz_path)

            mr_prep_slices_data_list.append(
                [current_mr_patient_id, current_slice, current_slice_mr_nii_gz_path, current_slice_seg_nii_gz_path])

    all_mr_prep_volume_data_csv = pd.DataFrame(columns=["patient_id", "mr_path", "seg_path"],
                                               data=mr_prep_volume_data_list)
    all_mr_prep_volume_data_csv.to_csv(all_mr_prep_volume_data_csv_path, index=False)

    all_mr_prep_slices_data_csv = pd.DataFrame(columns=["patient_id", 'slice_idx', "mr_path", "seg_path"],
                                               data=mr_prep_slices_data_list)
    all_mr_prep_slices_data_csv.to_csv(all_mr_prep_slices_data_csv_path, index=False)









