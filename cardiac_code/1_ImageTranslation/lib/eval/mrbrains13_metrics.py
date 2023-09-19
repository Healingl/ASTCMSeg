#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: Brain3DMRBrainS13
# @IDE: PyCharm
# @File: mrbrains13_metrics.py
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 20-11-19
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np
from lib.eval.binary_metric import dc, hd95, ravd



def mrbrains13_eval_metrics(gt,pred,class_label_list=[1,2,3],show=True):
    """
    0.Background (everything outside the brain)
    1.Cerebrospinal fluid (including ventricles)
    2.Gray matter (cortical gray matter and basal ganglia)
    3.White matter (including white matter lesions)
    :param gt: 3D array
    :param pred: 3D array
    :param class_label_list: [1,2,3]
    :return:
    """

    label_dict = {'1':'CSF','2':'GM','3':'WM'}

    pred = pred.astype(dtype='int')
    gt=gt.astype(dtype='int')

    dsc_dict = {}
    hd95_dict = {}
    avd_dict = {}
    for current_label in class_label_list:
        current_label = int(current_label)

        gt_c = np.zeros(gt.shape)
        y_c = np.zeros(gt.shape)
        gt_c[np.where(gt == current_label)]=1
        y_c[np.where(pred == current_label)]=1

        try:
            current_label_dsc = dc(y_c, gt_c)
        except:
            print('dc error gt:max %s, min %s, y_c:max %s, min %s' % (gt_c.max(), gt_c.min(), y_c.max(), y_c.min()))
            current_label_dsc = 0
        try:
            current_label_hd95 = hd95(y_c, gt_c)
        except:
            print('hd95 error gt:max %s, min %s, y_c:max %s, min %s' % (gt_c.max(), gt_c.min(), y_c.max(), y_c.min()))
            current_label_hd95 = 0
        try:
            current_label_avd = abs(ravd(y_c, gt_c))
        except:
            print('avd error gt:max %s, min %s, y_c:max %s, min %s'%(gt_c.max(),gt_c.min(),y_c.max(),y_c.min()))
            current_label_avd = 0

        dsc_dict['%s' % (label_dict[str(current_label)])] = round(current_label_dsc,4)
        hd95_dict['%s' % (label_dict[str(current_label)])] = round(current_label_hd95, 4)
        avd_dict['%s' % (label_dict[str(current_label)])] = round(current_label_avd, 4)
    if show:
        print('>>>'*30)
        print('DSC:',dsc_dict)
        print('HD95:',hd95_dict)
        print('AVD:',avd_dict)
        print('>>>'*30)
    return dsc_dict,hd95_dict,avd_dict