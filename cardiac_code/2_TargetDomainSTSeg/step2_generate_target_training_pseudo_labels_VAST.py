#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 23-2-13
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # #

import os
import torch
from torch.utils.data import DataLoader

from lib.utils.logging import *
from tqdm import tqdm
from lib.utils.simple_parser import Parser
import pandas as pd
import numpy as np
import cv2
import itertools
import shutil
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk


import random
import torch.backends.cudnn as cudnn
import matplotlib

matplotlib.use('Agg')
from tqdm import tqdm

from lib.dataloader.medical_loader_utils import get_order_crop_list
from lib.dataloader.medical_loader_utils import get_sample_area_by_centre, get_around_mask_order_sample_list
from lib.dataloader.medical_image_process import crop_cube_from_volume
from lib.utils.data_process import get_ND_bounding_box
from lib.dataloader.CMDA3DVolumeDataset import padding_array_by_3D_cropping_window, \
    reverse_padding_array_to_3D_origin_array
import time

def ias_thresh(conf_dict, num_classes, alpha, w=None, gamma=1.0):
    """

    :param conf_dict:
    :param num_classes:
    :param alpha:
    :param w: previous historical threshold
    :param gamma:
    :return:
    """
    if w is None:
        w = np.ones(num_classes)
    # threshold
    cls_thresh = np.ones(num_classes,dtype = np.float32)
    for idx_cls in np.arange(0, num_classes):
        if conf_dict[idx_cls] != None:
            arr = np.array(conf_dict[idx_cls])
            #  alpha=0.2
            cls_thresh[idx_cls] = np.percentile(arr, 100 * (1 - alpha * w[idx_cls] ** gamma))
    return cls_thresh

def sliding_window_inference(inference_img_array, current_models_list, current_yamls_list, val_crop_size, val_step_size, val_batch_size, cls_thresh):

    current_patient_pad_image_array, current_patient_padding_list = padding_array_by_3D_cropping_window(
        inference_img_array,
        val_crop_size,
        is_sample=True,
        constant_values=np.min(inference_img_array))

    full_vol_dim = current_patient_pad_image_array.shape

    # 3D
    model_preds = []
    for seg_model, current_model_yaml in zip(current_models_list, current_yamls_list):
        seg_model.cuda()  # go to gpu
        seg_model.eval()

        with torch.no_grad():
            # sliding windows
            if current_model_yaml['crop_type'] == 'random':
                sample_crop_list = get_order_crop_list(volume_shape=full_vol_dim,
                                                       crop_shape=val_crop_size,
                                                       extraction_step=val_step_size)
            elif current_model_yaml['crop_type'] == 'centre':
                current_brain_mask_array = np.zeros_like(current_patient_pad_image_array)
                current_brain_mask_array[current_patient_pad_image_array != 0] = 1
                idx_min, idx_max = get_ND_bounding_box(current_brain_mask_array, margin=0)

                current_centre_crop_min_point = get_sample_area_by_centre(mask_min_idx=idx_min,
                                                                          mask_max_idx=idx_max,
                                                                          full_vol_dim=full_vol_dim,
                                                                          crop_size=val_crop_size)
                around_tumor_order_sample_list = get_around_mask_order_sample_list(
                    origin_volume_size=full_vol_dim,
                    centre_crop_min_point=current_centre_crop_min_point,
                    crop_size=val_crop_size,
                    sample_sliding_step=val_step_size)
                sample_crop_list = around_tumor_order_sample_list
                # sample_crop_list = [current_centre_crop_min_point]
            else:
                assert False


            # (4,155,240,240)
            # prob array
            full_prob_np_array = np.zeros((current_model_yaml['num_classes'],
                                           full_vol_dim[0],
                                           full_vol_dim[1],
                                           full_vol_dim[2]))
            # count array
            full_count_np_array = np.zeros((current_model_yaml['num_classes'],
                                            full_vol_dim[0],
                                            full_vol_dim[1],
                                            full_vol_dim[2]))

            # batch_size
            PathNum = 0
            temp_crop_list = []
            temp_tensor_list = []

            for current_sample_idx, sample_crop in tqdm(enumerate(sample_crop_list), total=len(sample_crop_list),
                                                        ncols=50):
                # for current_sample_idx, sample_crop in enumerate(sample_crop_list):

                (current_crop_z_value, current_crop_y_value, current_crop_x_value) = sample_crop
                (crop_z_size, crop_y_size, crop_x_size) = val_crop_size

                current_crop_modal_array = crop_cube_from_volume(
                    origin_volume=current_patient_pad_image_array,
                    crop_point=sample_crop,
                    crop_size=val_crop_size
                )

                feature_np_array = np.array([current_crop_modal_array])
                feature_tensor = torch.unsqueeze(torch.from_numpy(feature_np_array).float(), dim=0)

                input_tensor = feature_tensor.cuda(non_blocking=True)

                # 为batch_size预测准备
                PathNum += 1
                temp_crop_list.append(sample_crop)
                temp_tensor_list.append(input_tensor)

                if PathNum == val_batch_size or current_sample_idx == len(sample_crop_list) - 1:
                    input_batch_tensor = torch.cat(temp_tensor_list, dim=0)
                    inputs = input_batch_tensor.cuda(non_blocking=True)
                    del input_batch_tensor

                    outputs = seg_model(inputs)
                    del inputs
                    # 转化成numpy
                    outputs_np = outputs.data.cpu().numpy()

                    for temp_crop_idx in range(len(temp_crop_list)):
                        temp_crop_z_value, temp_crop_y_value, temp_crop_x_value = temp_crop_list[temp_crop_idx]

                        # 获得小块, [4,64,64,64]
                        current_crop_prob_cube = outputs_np[temp_crop_idx]

                        assert len(current_crop_prob_cube) == current_model_yaml.num_classes

                        full_prob_np_array[:, temp_crop_z_value:temp_crop_z_value + crop_z_size,
                        temp_crop_y_value:temp_crop_y_value + crop_y_size,
                        temp_crop_x_value:temp_crop_x_value + crop_x_size] += current_crop_prob_cube[:, :, :, :]
                        full_count_np_array[:, temp_crop_z_value:temp_crop_z_value + crop_z_size,
                        temp_crop_y_value:temp_crop_y_value + crop_y_size,
                        temp_crop_x_value:temp_crop_x_value + crop_x_size] += 1

                    # 清空batch size
                    PathNum = 0
                    temp_crop_list = []
                    temp_tensor_list = []
                    torch.cuda.empty_cache()

                torch.cuda.empty_cache()

            # avoid no overlap region
            full_count_np_array[full_count_np_array == 0] = 1
            predict_seg_array = full_prob_np_array / full_count_np_array

            model_preds.append(predict_seg_array)
        seg_model.cpu()

    # model ensemble
    # (channel, slices, y, x)
    pred_prob_segs = np.mean(np.array(model_preds), axis=0)

    # (1, channel, slices, y, x)
    pred_batch_prob_segs = np.expand_dims(pred_prob_segs,axis=0)

    # ST pseudo label generation
    # torch
    y_pred_tensor = torch.from_numpy(pred_batch_prob_segs).type(torch.FloatTensor)

    #
    logits = F.softmax(y_pred_tensor, dim=1)

    # probs_pred, lbls_pred = logits.max(dim=1)
    max_items = logits.max(dim=1)
    logits_pred = max_items[0].data.cpu().numpy()
    label_pred = max_items[1].data.cpu().numpy()

    # # # # # # # # # # # # # # #
    # VAST pseudo label start
    # # # # # # # # # # # # # # #
    logits_cls_dict = {c: [cls_thresh[c]] for c in range(num_classes)}

    for cls in range(num_classes):
        logits_cls_dict[cls].extend(logits_pred[label_pred == cls].astype(np.float16))

    # instance adaptive selector
    tmp_cls_thresh = ias_thresh(logits_cls_dict, alpha=PSEUDO_PL_ALPHA, num_classes=num_classes, w=cls_thresh,
                                gamma=PSEUDO_PL_GAMMA)
    beta = PSEUDO_PL_BETA
    cls_thresh = beta * cls_thresh + (1 - beta) * tmp_cls_thresh
    cls_thresh[cls_thresh >= 1] = 0.999
    # # # # # # # # # # # # # # #
    # VAST pseudo label end
    # # # # # # # # # # # # # # #

    np_logits = logits.data.cpu().numpy()

    current_logits = np_logits[0]

    # save pseudo label
    # (z, y, x, channel)
    logit = current_logits.transpose(1, 2, 3, 0)
    # 按最大概率取标签
    pseudo_label = np.argmax(logit, axis=3)
    # 取标签通道中的最大概率值
    logit_amax = np.amax(logit, axis=3)

    print('thresh', cls_thresh)
    # apply_along_axis函数，其作用是沿着指定轴方向应用一个函数来处理数组的元素。在这里，apply_along_axis函数被用来将cls_thresh中的阈值应用到label数组的每一行上，生成一个包含相应阈值的新数组。
    label_cls_thresh = np.apply_along_axis(lambda x: [cls_thresh[e] for e in x], 1, pseudo_label)
    # # print('cls_thresh',cls_thresh)
    # print('label_cls_thresh', label_cls_thresh)
    ignore_index = logit_amax < label_cls_thresh
    pseudo_label[ignore_index] = 255

    # assert False
    current_prediction_volume = pseudo_label

    reverse_prediction_volume = reverse_padding_array_to_3D_origin_array(current_prediction_volume,
                                                                         current_patient_padding_list)

    patient_predict_seg_array = np.array(reverse_prediction_volume).astype(np.uint8)

    assert patient_predict_seg_array.shape == inference_img_array.shape

    return patient_predict_seg_array



if __name__ == "__main__":
    data_yaml_config = Parser('./config/data_config/data_settings_splitting.yaml')
    unlabeled_target_train_csv = pd.read_csv('./csv/train_test/ct_train_prep_volume.csv')

    ST_iteration = 5
    prep_target_plabel_data_dir = os.path.join(data_yaml_config['prep_target_VAST_plabel_data_dir'], 'Iter_%s'%(ST_iteration))
    if not os.path.exists(prep_target_plabel_data_dir): os.makedirs(prep_target_plabel_data_dir)
    print('pred save dir:',prep_target_plabel_data_dir)

    target_training_plabel_volume_csv_path= data_yaml_config['target_training_VAST_plabel_volume_csv_path']

    modality_and_model_weight_dict_list = [
        {
            'ModalName': 'real_t2',
            'Model_list': [
                # {
                #     'ModelName': 'CAMNet_Sup',
                #     'ModelConfig':
                #         {
                #             'model_yaml_path': './work_dir/CAMNet_Sup/CAMNet_Sup.yaml',
                #             'model_weight_path': './work_dir/CAMNet_Sup/model/epoch_62_dsc_0.7616_assd_3.1957/seg_unified_model.pth'
                #         },
                # },
                # {
                #     'ModelName': 'CAMNet_ST1',
                #     'ModelConfig':
                #         {
                #             'model_yaml_path': './work_dir/CAMNet_ST1/CAMNet_ST1.yaml',
                #             'model_weight_path': './work_dir/CAMNet_ST1/model/epoch_55_dsc_0.7905_assd_2.8157/seg_unified_model.pth'
                #         },
                # },
                # {
                #     'ModelName': 'CAMNet_ST2',
                #     'ModelConfig':
                #         {
                #             'model_yaml_path': './work_dir/CAMNet_ST2/CAMNet_ST2.yaml',
                #             'model_weight_path': './work_dir/CAMNet_ST2/model/epoch_65_dsc_0.7928_assd_2.819/seg_unified_model.pth'
                #         },
                # },
                # {
                #     'ModelName': 'CAMNet_ST3',
                #     'ModelConfig':
                #         {
                #             'model_yaml_path': './work_dir/CAMNet_ST3/CAMNet_ST3.yaml',
                #             'model_weight_path': './work_dir/CAMNet_ST3/model/epoch_70_dsc_0.805_assd_2.7237/seg_unified_model.pth'
                #         },
                # },


                {
                    'ModelName': 'CAMNet_ST4',
                    'ModelConfig':
                        {
                            'model_yaml_path': './work_dir/CAMNet_ST4/CAMNet_ST4.yaml',
                            'model_weight_path': './work_dir/CAMNet_ST4/model/epoch_48_dsc_0.8054_assd_3.0637/seg_unified_model.pth'
                        },
                },

            ]

        },
    ]




    #
    gpu_list = [0]
    random_seed = 2022
    val_batch_size = 1
    val_crop_size = [32, 256, 256]
    val_step_size = [16, 16, 16]

    PSEUDO_PL_GAMMA = 8.0
    PSEUDO_PL_BETA = 0.9
    PSEUDO_PL_ALPHA = 0.2

    confidence_thresh = 0.5

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # random seed and cuda
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.random.manual_seed(random_seed)
    random.seed(random_seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    softmax = lambda x: F.softmax(x, dim=1)

    torch.cuda.set_device('cuda:{}'.format(gpu_list[0]))

    for current_modality_and_models_dict in tqdm(modality_and_model_weight_dict_list, desc='modality'):
        current_modality_name = current_modality_and_models_dict['ModalName']

        current_model_dict_list = current_modality_and_models_dict['Model_list']

        current_models_list = []
        current_yamls_list = []
        num_classes = -1

        for current_model_dict in current_model_dict_list:
            current_model_name = current_model_dict['ModelName']
            current_model_yaml = Parser(current_model_dict['ModelConfig']['model_yaml_path'])
            current_model_weight_path = current_model_dict['ModelConfig']['model_weight_path']
            num_classes = current_model_yaml['num_classes']

            from lib.model.CAMNet import CAMNet
            basefilters = current_model_yaml['base_filter']
            seg_unified_model = CAMNet(in_channels=current_model_yaml['input_channels'],
                                       num_classes=current_model_yaml['num_classes'],
                                       kn=(basefilters * 2, basefilters * 3, basefilters * 4, basefilters * 5,
                                           basefilters * 6),
                                       ds=False,
                                       FMU='sub').cuda()

            # load weight
            seg_unified_model.load_state_dict(torch.load(current_model_weight_path, map_location='cpu'))

            current_models_list.append(seg_unified_model.cpu())
            current_yamls_list.append(current_model_yaml)



        #
        target_training_plabel_volume_list = []

        # VAST
        assert num_classes > 0


        cls_thresh = np.ones(num_classes) * confidence_thresh
        for idx, row in tqdm(unlabeled_target_train_csv.iterrows(), total=len(unlabeled_target_train_csv), ncols=50):
            # if idx > 1:
            #     break
            # input data
            current_patient_id = row['patient_id']

            # ct2mr
            current_patient_modal_path = row['ct_path']

            target_training_plabel_dir = os.path.join(prep_target_plabel_data_dir, str(current_patient_id))

            if not os.path.exists(target_training_plabel_dir): os.makedirs(target_training_plabel_dir)

            save_target_training_plabel_seg_nii_path = os.path.abspath( os.path.join(target_training_plabel_dir, '%s_prep_vol_seg.nii.gz' % (current_patient_id)))

            target_training_plabel_volume_list.append([current_patient_id, save_target_training_plabel_seg_nii_path])

            current_patient_modal_volume_array = sitk.GetArrayFromImage(sitk.ReadImage(current_patient_modal_path))

            # sliding window pred
            # pred and ensemble
            current_inference_img_array = current_patient_modal_volume_array

            current_prediction_volume = sliding_window_inference(inference_img_array=current_inference_img_array,
                                                                 current_models_list=current_models_list,
                                                                 current_yamls_list=current_yamls_list,
                                                                 val_crop_size=val_crop_size,
                                                                 val_step_size=val_step_size,
                                                                 val_batch_size=val_batch_size,
                                                                 cls_thresh = cls_thresh
                                                                 )

            current_patient_predict_seg_array = np.array(current_prediction_volume).astype(np.uint8)

            assert current_patient_modal_volume_array.shape == current_patient_predict_seg_array.shape

            res_lb_o = sitk.GetImageFromArray(current_patient_predict_seg_array)
            sitk.WriteImage(res_lb_o, save_target_training_plabel_seg_nii_path)



        target_training_plabel_volume_csv = pd.DataFrame(columns=["id", "seg_path"],
                                                   data=target_training_plabel_volume_list)
        target_training_plabel_volume_csv.to_csv(target_training_plabel_volume_csv_path, index=False)