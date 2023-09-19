#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: Brain3DISEG17
# @IDE: PyCharm
# @File: iseg17_metrics.py
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 20-12-5
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np
from lib.eval.binary_metric import dc, jc

def brainptm_eval_metrics(gt,pred,class_label_list=[1,2,3,4],show=True):
    """
    {'bg':0, 'cst_left_seg':1,  'cst_right_seg':2,  'or_left_seg':3,  'or_right_seg':4}
    0.Background (everything outside the brain)
    1.cst_left_seg
    2.cst_right_seg
    3.or_left_seg
    4.or_right_seg
    :param gt: 3D array
    :param pred: 3D array
    :param class_label_list: [1,2,3]
    :return:
    """

    label_dict = {'1':'cst_left_seg','2':'cst_right_seg','3':'or_left_seg','4':'or_right_seg'}

    pred = pred.astype(dtype='int')
    gt=gt.astype(dtype='int')

    dsc_dict = {}
    jc_dict = {}

    for current_label in class_label_list:
        current_label = int(current_label)

        gt_c = np.zeros(gt.shape)
        y_c = np.zeros(gt.shape)
        gt_c[np.where(gt==current_label)]=1
        y_c[np.where(pred==current_label)]=1

        try:
            current_label_dsc = dc(y_c,gt_c)
        except:
            print('dc error gt:max %s, min %s, y_c:max %s, min %s' % (gt_c.max(), gt_c.min(), y_c.max(), y_c.min()))
            current_label_dsc = 0
        try:
            current_label_jc = jc(y_c,gt_c)
        except:
            print('jc error gt:max %s, min %s, y_c:max %s, min %s' % (gt_c.max(), gt_c.min(), y_c.max(), y_c.min()))
            current_label_jc = 0


        dsc_dict['%s' % (label_dict[str(current_label)])] = round(current_label_dsc,4)
        jc_dict['%s' % (label_dict[str(current_label)])] = round(current_label_jc,4)

    if False:
        print('>>>'*30)
        print('DSC:',dsc_dict)
        print('Jaccard coefficient:',jc_dict)
        print('>>>'*30)
    return dsc_dict,jc_dict

def brainptm_eval_metrics_single_model(gt, pred, region_name, class_label_list=[1], show=True):
    """
    if self.region_name == 'brainptm_cst_left':
            img_seg = img_cst_left_seg
        elif self.region_name == 'brainptm_cst_right':
            img_seg = img_cst_right_seg
        elif self.region_name == 'brainptm_or_left':
            img_seg = img_or_left_seg
        elif self.region_name == 'brainptm_or_right':
            img_seg = img_or_right_seg
        else:
            assert False
    :param gt: 3D array
    :param pred: 3D array
    :param class_label_list: [1,2,3]
    :return:
    """
    assert region_name in ['brainptm_cst_left', 'brainptm_cst_right','brainptm_or_left', 'brainptm_or_right']

    label_dict = {'1':region_name}

    pred = pred.astype(dtype='int')
    gt=gt.astype(dtype='int')

    dsc_dict = {}
    jc_dict = {}

    for current_label in class_label_list:
        current_label = int(current_label)

        gt_c = np.zeros(gt.shape)
        y_c = np.zeros(gt.shape)
        gt_c[np.where(gt==current_label)]=1
        y_c[np.where(pred==current_label)]=1

        try:
            current_label_dsc = dc(y_c,gt_c)
        except:
            print('dc error gt:max %s, min %s, y_c:max %s, min %s' % (gt_c.max(), gt_c.min(), y_c.max(), y_c.min()))
            current_label_dsc = 0
        try:
            current_label_jc = jc(y_c,gt_c)
        except:
            print('jc error gt:max %s, min %s, y_c:max %s, min %s' % (gt_c.max(), gt_c.min(), y_c.max(), y_c.min()))
            current_label_jc = 0


        dsc_dict['%s' % (label_dict[str(current_label)])] = round(current_label_dsc,4)
        jc_dict['%s' % (label_dict[str(current_label)])] = round(current_label_jc,4)

    if False:
        print('>>>'*30)
        print('DSC:',dsc_dict)
        print('Jaccard coefficient:',jc_dict)
        print('>>>'*30)
    return dsc_dict,jc_dict