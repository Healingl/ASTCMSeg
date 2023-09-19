#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable



class CrossEntropy2d(nn.Module):

    def __init__(self):
        super(CrossEntropy2d, self).__init__()

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """

        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()


        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()

        predict = predict.view(-1, c)
        target = target.view(-1)

        criterion = nn.CrossEntropyLoss()
        # print(predict.dtype,target.dtype)
        loss = criterion(predict.float(), target.long())

        return loss

