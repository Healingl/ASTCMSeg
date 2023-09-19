#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import torch
from torch.utils.data import DataLoader


from torch.optim.lr_scheduler import StepLR

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
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler

from lib.dataloader.CMDA3DVolumeDataset import CMDA3DVolumeDataset
from lib.dataloader.CMDA3DVolumeDataset import padding_array_by_3D_cropping_window, \
    reverse_padding_array_to_3D_origin_array

from lib.dataloader.medical_image_process import braincmda2022_cyclegan_normalization, braincmda2022_minmax_normalization
from lib.dataloader.medical_image_process import braincmda2022_onezero_to_meanstd_normalization, braincmda2022_meanstd_to_onezero_normalization

from lib.dataloader.medical_loader_utils import get_order_crop_list
from lib.dataloader.medical_loader_utils import get_sample_area_by_centre, get_around_mask_order_sample_list
from lib.dataloader.medical_image_process import crop_cube_from_volume
from lib.utils.data_process import get_ND_bounding_box

import SimpleITK as sitk

import random
import matplotlib

matplotlib.use('Agg')
from tqdm import tqdm

from lib.model.save_and_load.utils import save_models, load_models
import time

current_time = str(time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time())))

from lib.eval.cardiac_unpaired_metric import cardiac_unpaired_metric



def validate_by_real_test():
    """

    :param seg_model:
    :param epoch:
    :return:
    """
    val_modal_volume_csv = pd.read_csv(model_yaml_config['val_real_t2_volume_csv_path'])
    cal_class_list = model_yaml_config['cal_class_list']
    label_to_cate_dict = model_yaml_config['label_to_cate_dict']
    val_batch_size = model_yaml_config['val_batch_size']
    val_crop_size = model_yaml_config['crop_size']
    val_step_size = model_yaml_config['val_slide_step_size']

    seg_unified_model.eval()
    with torch.no_grad():

        eval_dsc_results = {}
        eval_assd_results = {}

        for current_cat in cal_class_list:
            eval_dsc_results[str(label_to_cate_dict[str(current_cat)])] = []
            eval_assd_results[str(label_to_cate_dict[str(current_cat)])] = []

        for idx, row in tqdm(val_modal_volume_csv.iterrows(), total=len(val_modal_volume_csv), ncols=50):
            # if idx > 21:
            #     break
            # input data
            current_patient_id = row['patient_id']

            current_patient_modal_path = row['ct_path']
            # label
            current_patient_seg_path = row['seg_path']


            current_patient_modal_volume_array = sitk.GetArrayFromImage(sitk.ReadImage(current_patient_modal_path))
            # current_patient_modal_volume_array = braincmda2022_meanstd_to_onezero_normalization(current_patient_modal_volume_array)

            current_patient_seg_volume_array = sitk.GetArrayFromImage(sitk.ReadImage(current_patient_seg_path))

            # sliding window pred
            current_patient_pad_image_array, current_patient_padding_list = padding_array_by_3D_cropping_window(
                current_patient_modal_volume_array,
                val_crop_size,
                is_sample=True,
                constant_values=np.min(current_patient_modal_volume_array))
            full_vol_dim = current_patient_pad_image_array.shape

            # sliding windows
            if model_yaml_config['crop_type'] == 'random':
                sample_crop_list = get_order_crop_list(volume_shape=full_vol_dim,
                                                       crop_shape=val_crop_size,
                                                       extraction_step=val_step_size)
            else:
                assert False


            # (4,155,240,240)
            # prob array
            full_prob_np_array = np.zeros((model_yaml_config['num_classes'],
                                           full_vol_dim[0],
                                           full_vol_dim[1],
                                           full_vol_dim[2]))
            # count array
            full_count_np_array = np.zeros((model_yaml_config['num_classes'],
                                            full_vol_dim[0],
                                            full_vol_dim[1],
                                            full_vol_dim[2]))

            # batch_size
            PathNum = 0
            temp_crop_list = []
            temp_tensor_list = []

            # for current_sample_idx, sample_crop in tqdm(enumerate(sample_crop_list), total=len(sample_crop_list),ncols=50):
            for current_sample_idx, sample_crop in enumerate(sample_crop_list):

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

                    outputs = seg_unified_model(inputs)
                    del inputs
                    # 转化成numpy
                    outputs_np = outputs.data.cpu().numpy()

                    for temp_crop_idx in range(len(temp_crop_list)):
                        temp_crop_z_value, temp_crop_y_value, temp_crop_x_value = temp_crop_list[temp_crop_idx]

                        # 获得小块, [4,64,64,64]
                        current_crop_prob_cube = outputs_np[temp_crop_idx]

                        assert len(current_crop_prob_cube) == model_yaml_config.num_classes

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

            # (4,155,240,240) -> (155,240,240)
            current_prediction_volume = np.argmax(predict_seg_array, axis=0)

            reverse_prediction_volume = reverse_padding_array_to_3D_origin_array(current_prediction_volume,
                                                                                 current_patient_padding_list)

            patient_predict_seg_array = np.array(reverse_prediction_volume).astype(np.uint8)

            assert current_patient_seg_volume_array.shape == patient_predict_seg_array.shape

            current_patient_dsc_dict,current_patient_assd_dict = cardiac_unpaired_metric(gt=current_patient_seg_volume_array, pred=patient_predict_seg_array,
                                                               class_label_list=cal_class_list)

            for key, value in current_patient_dsc_dict.items():
                eval_dsc_results[str(key)].append(round(value,4))

            for key, value in current_patient_assd_dict.items():
                eval_assd_results[str(key)].append(round(value, 4))

        eval_dsc_dict = {}
        eval_dsc_list = []
        for key, value in eval_dsc_results.items():
            current_mean_dsc = round(np.array(value).mean(),4)
            eval_dsc_dict['%s_dsc'%(key)] = current_mean_dsc
            eval_dsc_list.append(current_mean_dsc)
        mean_dsc_value = round(sum(eval_dsc_list)/len(eval_dsc_list),4)

        eval_assd_dict = {}
        eval_assd_list = []
        for key, value in eval_assd_results.items():
            current_mean_assd = round(np.array(value).mean(), 4)
            eval_assd_dict['%s_assd' % (key)] = current_mean_assd
            eval_assd_list.append(current_mean_assd)
        mean_assd_value = round(sum(eval_assd_list) / len(eval_assd_list), 4)

        return mean_dsc_value, eval_dsc_dict, mean_assd_value, eval_assd_dict,



if __name__ == "__main__":
    model_config_path = './config/model_config/CAMNet_ST5.yaml'

    model_yaml_config = Parser(model_config_path)

    model_name = model_yaml_config['model_name']
    workdir = model_yaml_config['workdir']
    if not os.path.exists(workdir): os.makedirs(workdir)

    logger = get_logger(os.path.join(workdir, '%s.log' % (model_name)))
    shutil.copy(model_config_path, os.path.join(workdir, os.path.basename(model_config_path)))

    logger.info(str(model_yaml_config))

    gpu_list = model_yaml_config['gpus']
    train_labeled_batch_size = model_yaml_config['train_labeled_batch_size']
    train_unlabeled_batch_size = model_yaml_config['train_unlabeled_batch_size']

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 加载数据集
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    np.random.seed(model_yaml_config['seed'])
    random.seed(model_yaml_config['seed'])
    torch.cuda.manual_seed_all(model_yaml_config['seed'])
    torch.random.manual_seed(model_yaml_config['seed'])
    random.seed(model_yaml_config['seed'])


    # unpaired source fake_t2 and target t2 dataset
    # train
    train_labeled_fake_t2_volume_dataset = CMDA3DVolumeDataset(
        sample_3d_volume_csv_path=model_yaml_config['labeled_fake_t2_volume_csv_path'],
        mode='train',
        annotation_type = 'labeled',
        modality='fake_t2',
        data_num=-1,
        use_aug=model_yaml_config['use_aug'],
        crop_size=model_yaml_config['crop_size'],
        step_size=model_yaml_config['val_slide_step_size'],
        crop_type=model_yaml_config['crop_type'],
        prep_target_plabel_data_dir=model_yaml_config['gen_target_plabel_data_dir']
    )
    train_unlabeled_real_t2_volume_dataset = CMDA3DVolumeDataset(
        sample_3d_volume_csv_path=model_yaml_config['unlabeled_t2_volume_csv_path'],
        mode='train',
        annotation_type='labeled',
        modality='t2',
        data_num=-1,
        use_aug=model_yaml_config['use_aug'],
        crop_size=model_yaml_config['crop_size'],
        step_size=model_yaml_config['val_slide_step_size'],
        crop_type=model_yaml_config['crop_type'],
        prep_target_plabel_data_dir=model_yaml_config['gen_target_plabel_data_dir']
    )

    train_labeled_fake_t2_volume_loader = DataLoader(train_labeled_fake_t2_volume_dataset,
                                                     shuffle=True,
                                                     pin_memory=True,
                                                     batch_size=model_yaml_config['train_labeled_batch_size'],
                                                     num_workers=1)
    train_unlabeled_real_t2_volume_loader = DataLoader(train_unlabeled_real_t2_volume_dataset,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       batch_size=model_yaml_config['train_unlabeled_batch_size'],
                                                       num_workers=1)

    logger.info('train_labeled_fake_t2_volume_loader: %s' % (len(train_labeled_fake_t2_volume_loader)))
    logger.info('train_unlabeled_real_t2_volume_loader: %s' % (len(train_unlabeled_real_t2_volume_loader)))

    torch.cuda.set_device('cuda:{}'.format(gpu_list[0]))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 训练开始
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    MODEL_DIR = os.path.join(model_yaml_config['workdir'], 'model')
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    logger.info('building models ...')
    assert model_yaml_config['num_classes'] == len(model_yaml_config['cate_to_label_dict']), 'Error Class Num: %s' % (
        model_yaml_config['num_classes'])

    from lib.model.CAMNet import CAMNet
    basefilters = model_yaml_config['base_filter']
    seg_unified_model = CAMNet(in_channels=model_yaml_config['input_channels'],num_classes=model_yaml_config['num_classes'],
                               kn=(basefilters*2, basefilters*3, basefilters*4, basefilters*5, basefilters*6),
                               ds=False,
                               FMU='sub').cuda()



    # Load or Save Model
    model_dict = {}
    model_dict['seg_unified_model'] = seg_unified_model

    if model_yaml_config['resume'] and os.path.exists(model_yaml_config['resume_weight_dir']):
        logger.info('>>>' * 30)
        logger.info('loading pretrained weight:%s' % (model_yaml_config['resume_weight_dir']))
        logger.info('>>>' * 30)

        load_models(model_dict, model_yaml_config['resume_weight_dir'])

    # optimizers
    seg_unified_model_opt = torch.optim.Adam(seg_unified_model.parameters(),
                                             lr=model_yaml_config['seg_lr'],
                                             weight_decay=model_yaml_config['weight_decay'],
                                             betas=model_yaml_config['seg_betas'])

    # Learning rate update schedulers
    lr_scheduler_seg_unified = torch.optim.lr_scheduler.CosineAnnealingLR(seg_unified_model_opt,
                                                                          T_max=model_yaml_config['num_epochs'],
                                                                          eta_min=model_yaml_config['eta_min'],
                                                                          last_epoch=-1, )



    from lib.loss.BCEDice import BCEDiceLoss
    bce_dice_loss = BCEDiceLoss(n_classes=model_yaml_config['num_classes'], do_softmax=True)
    from torch.nn.modules.loss import CrossEntropyLoss
    self_training_ce_loss = CrossEntropyLoss(ignore_index=255, reduction='mean')
    # self_training_ce_loss = CrossEntropyLoss( reduction='mean')
    # # # # # # # # # # # # # # # # # # # # # # # # #
    # # Training
    # # # # # # # # # # # # # # # # # # # # # # # # #

    current_epoch = 0
    best_dice = 0
    best_epoch = 0
    iter_num = 0

    # create grad scaler
    scaler = GradScaler()

    for current_epoch in range(model_yaml_config['num_epochs']):
        logger.info('>>>>' * 30)
        logger.info('[Epoch %s/%s]' % (current_epoch, model_yaml_config['num_epochs']))

        train_labeled_fake_t2_volume_loader_iter = iter(train_labeled_fake_t2_volume_loader)
        train_unlabeled_real_t2_volume_loader_iter = iter(train_unlabeled_real_t2_volume_loader)
        epoch_all_iter_num = len(train_labeled_fake_t2_volume_loader)

        seg_unified_model.train()
        for current_epoch_iter in tqdm(range(epoch_all_iter_num), total=epoch_all_iter_num):
            # if current_epoch_iter > 5:
            #     break
            # # # # # # # # # # # # # # # # # # # # # # # # #
            # # Getting tensor
            # # # # # # # # # # # # # # # # # # # # # # # # #

            for data_idx in [0,1]:
                if data_idx == 0:
                    # source fake_t2
                    labeled_t2_feature_tensor, labeled_t2_seg_gt_tensor = next(train_labeled_fake_t2_volume_loader_iter)
                else:
                    try:
                        # real t2
                        labeled_t2_feature_tensor, labeled_t2_seg_gt_tensor = next(train_unlabeled_real_t2_volume_loader_iter)
                    except StopIteration:
                        train_unlabeled_real_t2_volume_loader_iter = iter(train_unlabeled_real_t2_volume_loader)
                        labeled_t2_feature_tensor, labeled_t2_seg_gt_tensor = next(train_unlabeled_real_t2_volume_loader_iter)

                # source
                labeled_t2_feature_tensor = labeled_t2_feature_tensor.cuda(non_blocking=True)
                labeled_t2_seg_gt_tensor = labeled_t2_seg_gt_tensor.type(torch.LongTensor).cuda(non_blocking=True)

                lbvs = np.unique(labeled_t2_seg_gt_tensor.cpu().numpy())
                if len(lbvs) < 2:
                    logger.info('[Epoch %d / %d], [iter %d / %d], [No label skip %s]' % (
                        current_epoch,
                        model_yaml_config['num_epochs'],
                        epoch_all_iter_num,
                        current_epoch_iter,
                        str(lbvs)
                    ))
                    continue

                # # # # # # # # # # # # # # # # # # # # # # # # #
                # # Supervised Seg Loss
                # # # # # # # # # # # # # # # # # # # # # # # # #
                #


                # zero grade
                seg_unified_model_opt.zero_grad()

                with autocast(enabled=model_yaml_config['fp16']):
                    # output
                    labeled_t2_pred_outputs = seg_unified_model(labeled_t2_feature_tensor)

                    if data_idx == 0:
                        # ==== supervised loss ====
                        loss_t2_sg_1 = bce_dice_loss(labeled_t2_pred_outputs, labeled_t2_seg_gt_tensor)

                        total_loss = loss_t2_sg_1
                    else:
                        # ==== semi-supervised loss ====
                        loss_t2_st = self_training_ce_loss(labeled_t2_pred_outputs,labeled_t2_seg_gt_tensor)
                        total_loss = loss_t2_st


                scaler.scale(total_loss).backward()
                scaler.step(seg_unified_model_opt)
                scaler.update()

                # total_loss.backward()
                # seg_unified_model_opt.step()


                iter_num = iter_num + 1
                loss_all_log = round(total_loss.item(), 4)
                # semi_log = '%s, %s'%(round(loss_semi_seg.item(), 4), round(consistency_weight, 4))
                semi_log = ''
                seg_lr = seg_unified_model_opt.param_groups[0]['lr']
                logger.info('[Epoch %d / %d], [iter %d / %d], [seg_lr: %.7f], [seg loss: %s], [semi loss: %s]' % ( current_epoch,
                                                                                                                   model_yaml_config['num_epochs'],
                                                                                                                   epoch_all_iter_num,
                                                                                                                   current_epoch_iter,
                                                                                                                   seg_lr,
                                                                                                                   loss_all_log,
                                                                                                                   semi_log))

        # Update learning rates
        lr_scheduler_seg_unified.step()

        # Eval
        from torchvision.utils import save_image, make_grid
        if current_epoch % model_yaml_config['eval_metric_epoch'] == 0:
            logger.info('>>>>' * 30)
            logger.info('Evaluate Real Test: %s'%(model_yaml_config['val_real_t2_volume_csv_path']))
            mean_dsc_value, eval_dsc_dict,mean_assd_value, eval_assd_dict = validate_by_real_test()
            logger.info('mean_dsc_value: %s, eval_dsc_dict: %s' % (mean_dsc_value, str(eval_dsc_dict)))
            logger.info('mean_assd_value: %s, eval_assd_dict: %s' % (mean_assd_value, str(eval_assd_dict)))

            if mean_dsc_value > best_dice:
                best_epoch = current_epoch
                best_dice = mean_dsc_value
                save_models(model_dict, MODEL_DIR + '/epoch_%s_dsc_%s_assd_%s/' % (best_epoch, best_dice, mean_assd_value))

                logger.info('Get New Best Dice! Best Epoch: %06d : Best_Dice %.4f ' % (best_epoch, best_dice))
                logger.info('>>>>' * 30)
            logger.info('>>>>' * 30)
