#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: BrainTS2020SegDeepSupTorch16
# @IDE: PyCharm
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 2021/12/27
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

class BCEDiceLoss(nn.Module):
    """
    BCEWithLogitsLoss+Dice
    """

    def __init__(self, n_classes=16, do_softmax=True):
        super(BCEDiceLoss, self).__init__()
        self.do_softmax = do_softmax
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def compute_intersection(self, inputs, targets):

        intersection = torch.sum(inputs * targets)

        return intersection

    def metric_dice_compute(self, inputs_logits, targets, smooth = 1e-6):
        input_onehot = inputs_logits > 0.5
        input_onehot = input_onehot.float()

        metric_dice = (2*torch.sum(input_onehot * targets)+smooth) / (input_onehot.sum() + targets.sum() + smooth)

        return metric_dice.data.cpu().numpy()

    def binary_dice_loss(self, inputs, targets, smooth = 1e-6, pow = False):

        intersection = self.compute_intersection(inputs, targets)
        dice_loss = (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        if pow:
            dice_loss = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)

        dice_loss = 1 - dice_loss
        return dice_loss


    def bce_loss(self, inputs, targets):
        targets = targets.float()
        bce_criterion = nn.BCEWithLogitsLoss()
        bce_loss = bce_criterion(input=inputs, target=targets)

        return bce_loss

    def forward(self, inputs, target):
        if self.do_softmax:
            inputs = torch.softmax(inputs, dim=1)
            # print(inputs)
            # print(self.nomalization(inputs)[0][0][0])
            # print(">>>>>" * 20)
            # print(torch.softmax(inputs)[0][0][0])
            # print(">>>>>"*20)
            # print(F.softmax(inputs)[0][0][0])
            # assert False

        target = self._one_hot_encoder(target.unsqueeze(1))
        b,c,z,y,x = target.size()

        dice_loss_sum = 0
        for i in range(c):
            current_channel_dice_loss = self.binary_dice_loss(inputs[:, i, ...], target[:, i, ...], pow=True)
            dice_loss_sum = dice_loss_sum + current_channel_dice_loss

        mean_dice_loss = dice_loss_sum / c

        bce_loss = self.bce_loss(inputs=inputs, targets=target)
        combo_loss = 0.5*mean_dice_loss + 0.5*bce_loss
        return combo_loss




if __name__ == "__main__":
    # batch_size, tumor_regions, dim_z, dim_y, dim_x
    num_classes = 16
    # 0~1 prob
    y_pred_shape = (2, num_classes, 8, 8, 8)
    # int type
    y_gt_shape = (2, 8, 8, 8)


    # softmax: neg 0 pos 1 zero 0.5
    y_pred = np.random.uniform(low=-1.0, high=1.0, size=y_pred_shape)
    y_gt = np.random.randint(low=0, high=16, size=y_gt_shape)


    # print(y_pred.shape, y_gt.shape)
    # print(y_pred[0][0][0])
    # print(y_gt[0][0][0])

    # y_pred_tensor = torch.from_numpy(y_pred).type(torch.FloatTensor)
    # y_gt_tensor = torch.from_numpy(y_gt).type(torch.FloatTensor)

    y_pred_tensor = torch.from_numpy(y_pred).type(torch.FloatTensor)
    y_gt_tensor = torch.from_numpy(y_gt).type(torch.LongTensor)



    # binary ce
    criterion = BCEDiceLoss(n_classes=num_classes, do_softmax=True)
    # metric = criterion.metric
    # print(y_pred_tensor>0.5)
    # assert False
    loss = criterion(inputs=y_pred_tensor, target=y_gt_tensor)
    # loss = criterion(inputs=y_pred_tensor, target=y_gt_tensor)
    print('loss', loss.item())

    # metric_ = metric(inputs=y_gt_tensor, target=y_gt_tensor)
    # print(metric_)