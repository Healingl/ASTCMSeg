#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 22-4-29
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from lib.eval.binary_metric import dc, assd, jc, asd

def cardiac_unpaired_metric(gt, pred, class_label_list=[1,2,3,4]):
    """
     {'0':'bg', '1':'MYO',  '2':'LAC', '3':'LVC', '4':'AA'}
    dc, assd
    :param gt: 3D array
    :param pred: 3D array
    :param class_label_list: [1,2,3]
    :return:
    """

    label_dict =  {'1':'MYO',  '2':'LAC', '3':'LVC', '4':'AA'}

    pred = pred.astype(dtype='int')
    gt=gt.astype(dtype='int')

    dsc_dict = {}
    assd_dict = {}

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
            current_label_assd = assd(y_c, gt_c, voxelspacing=(1,1,1))
        except:
            current_label_assd = 0

        dsc_dict['%s' % (label_dict[str(current_label)])] = round(current_label_dsc,4)
        assd_dict['%s' % (label_dict[str(current_label)])] = round(current_label_assd,4)

    return dsc_dict, assd_dict
