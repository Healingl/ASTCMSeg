#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @Project: BrainTissueISEG2017
# @IDE: PyCharm
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 2022/4/18
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn

from lib.model.basic_module import *

# Anisotropic Context-aware Feature Fusion Module
class ContextBlock3D(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio=8,
                 pooling_type='att',
                 fusion_types=('channel_mul',)):

        super(ContextBlock3D, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        #  input feats
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes // ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            # inplanes = input features -》 1 channel
            self.conv_mask = nn.Conv3d(inplanes, 1, kernel_size=(1,1,1))
            self.softmax = nn.Softmax(dim=2)
        # else:
        #     self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # if 'channel_add' in fusion_types:
        #     self.channel_add_conv = nn.Sequential(
        #         nn.Conv3d(self.inplanes, self.planes, kernel_size=(1,1,1)),
        #         nn.LayerNorm([self.planes, 1, 1, 1]),
        #         nn.ReLU(inplace=True),  # yapf: disable
        #         nn.Conv3d(self.planes, self.inplanes, kernel_size=(1,1,1)))
        # else:
        #     self.channel_add_conv = None

        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv3d(self.inplanes, self.planes, kernel_size=(1,1,1)),
                nn.LayerNorm([self.planes, 1, 1, 1]),
                nn.LeakyReLU(inplace=True),  # yapf: disable
                nn.Conv3d(self.planes, self.inplanes, kernel_size=(1,1,1)))
        else:
            self.channel_mul_conv = None


    def spatial_pool(self, x):
        batch, channel, depth, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, D *  H * W]
            input_x = input_x.view(batch, channel, depth * height * width)
            # [N, 1, C, D * H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, D, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, D * H * W]
            context_mask = context_mask.view(batch, 1, depth * height * width)
            # [N, 1, D * H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1,D * H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # print('context_mask',context_mask.shape)
            # [N, 1, C, 1, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1, 1]
            context = context.view(batch, channel, 1, 1, 1)
        else:
            assert False
        # else:
        #     # [N, C, 1, 1, 1]
        #     context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        context = self.channel_mul_conv(context)



        return context

class AFF3D(nn.Module):
    '''
    Anisotropic Context-aware Feature Fusion Module
    '''

    def __init__(self, channels, r=4):
        super(AFF3D, self).__init__()
        self.global_context = ContextBlock3D(inplanes=channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual

        wei = self.sigmoid(self.global_context(xa))

        xo = x * wei + residual * (1 - wei)
        return xo


class ADD3D(nn.Module):
    '''
    Anisotropic Context-aware Feature Fusion Module
    '''

    def __init__(self, channels, r=4):
        super(ADD3D, self).__init__()
        inter_channels = int(channels // r)


    def forward(self, x, residual):
        # xa = x + residual
        xa = torch.abs(x - residual)
        return xa

def FMU(x1, x2, mode='sub'):
    """
    feature merging unit
    Args:
        x1:
        x2:
        mode: type of fusion
    Returns:
    """
    if mode == 'sum':
        return torch.add(x1, x2)
    elif mode == 'sub':
        return torch.abs(x1 - x2)
    elif mode == 'cat':
        return torch.cat((x1, x2), dim=1)
    else:
        raise Exception('Unexpected mode')

class ContextDown(BasicNet):
    def __init__(self, in_channels, out_channels, mode: tuple, FMU='context', downsample=True, min_z=8):
        """
        basic module at downsampling stage
        Args:
            in_channels:
            out_channels:
            mode: represent the streams coming in and out. e.g., ('2d', 'both'): one input stream (2d) and two output streams (2d and 3d)
            FMU: determine the type of feature fusion if there are two input streams
            downsample: determine whether to downsample input features (only the first module of MNet do not downsample)
            min_z: if the size of z-axis < min_z, maxpooling won't be applied along z-axis
        """
        super().__init__()

        # 输入模式， 输出模式
        self.mode_in, self.mode_out = mode

        self.downsample = downsample
        # self.context_feature_fusion = ADD3D(channels=in_channels)
        self.context_feature_fusion = AFF3D(channels=in_channels)

        self.min_z = min_z
        norm_args = (self.norm_kwargs, self.norm_kwargs)
        activation_args = (self.activation_kwargs, self.activation_kwargs)

        if self.mode_out == '2d' or self.mode_out == 'both':
            self.CB2d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

        if self.mode_out == '3d' or self.mode_out == 'both':
            self.CB3d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

    def forward(self, x):
        if self.downsample:
            if self.mode_in == 'both':
                # split
                x2d, x3d = x
                p2d = F.max_pool3d(x2d, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                if x3d.shape[2] >= self.min_z:
                    p3d = F.max_pool3d(x3d, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    p3d = F.max_pool3d(x3d, kernel_size=(1, 2, 2), stride=(1, 2, 2))

                x = self.context_feature_fusion(p2d, p3d)

            elif self.mode_in == '2d':
                x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

            elif self.mode_in == '3d':
                if x.shape[2] >= self.min_z:
                    x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                pass

        if self.mode_out == '2d':
            return self.CB2d(x)
        elif self.mode_out == '3d':
            return self.CB3d(x)
        elif self.mode_out == 'both':
            return self.CB2d(x), self.CB3d(x)

class ContextUp(BasicNet):
    def __init__(self, in_channels, out_channels, up_channels: tuple, mode: tuple, FMU='sub'):
        """
        basic module at upsampling stage
        Args:
            in_channels:
            out_channels:
            mode: represent the streams coming in and out. e.g., ('2d', 'both'): one input stream (2d) and two output streams (2d and 3d)
            FMU: determine the type of feature fusion if there are two input streams
        """
        super().__init__()
        self.mode_in, self.mode_out = mode

        # self.context_fusion_skip_1 = ADD3D(channels=up_channels[0])
        # self.context_fusion_skip_2 = ADD3D(channels=up_channels[1])
        self.context_fusion_skip_1 = AFF3D(channels=up_channels[0])
        self.context_fusion_skip_2 = AFF3D(channels=up_channels[1])

        norm_args = (self.norm_kwargs, self.norm_kwargs)
        activation_args = (self.activation_kwargs, self.activation_kwargs)

        if self.mode_out == '2d' or self.mode_out == 'both':
            self.CB2d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

        if self.mode_out == '3d' or self.mode_out == 'both':
            self.CB3d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

    def forward(self, x):
        x2d, xskip2d, x3d, xskip3d = x

        tarSize = xskip2d.shape[2:]

        up2d = F.interpolate(x2d, size=tarSize, mode='trilinear', align_corners=False)
        up3d = F.interpolate(x3d, size=tarSize, mode='trilinear', align_corners=False)

        # xskip2d, xskip3d: high resolution,
        # up2d, up3d: rescale resolution

        cat = torch.cat([self.context_fusion_skip_1(xskip2d, xskip3d), self.context_fusion_skip_2(up2d, up3d)], dim=1)

        if self.mode_out == '2d':
            return self.CB2d(cat)
        elif self.mode_out == '3d':
            return self.CB3d(cat)
        elif self.mode_out == 'both':
            return self.CB2d(cat), self.CB3d(cat)

class Down(BasicNet):
    def __init__(self, in_channels, out_channels, mode: tuple, FMU='sub', downsample=True, min_z=8):
        """
        basic module at downsampling stage
        Args:
            in_channels:
            out_channels:
            mode: represent the streams coming in and out. e.g., ('2d', 'both'): one input stream (2d) and two output streams (2d and 3d)
            FMU: determine the type of feature fusion if there are two input streams
            downsample: determine whether to downsample input features (only the first module of MNet do not downsample)
            min_z: if the size of z-axis < min_z, maxpooling won't be applied along z-axis
        """
        super().__init__()
        self.mode_in, self.mode_out = mode
        self.downsample = downsample
        self.FMU = FMU
        self.min_z = min_z
        norm_args = (self.norm_kwargs, self.norm_kwargs)
        activation_args = (self.activation_kwargs, self.activation_kwargs)

        if self.mode_out == '2d' or self.mode_out == 'both':
            self.CB2d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

        if self.mode_out == '3d' or self.mode_out == 'both':
            self.CB3d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

    def forward(self, x):
        if self.downsample:
            if self.mode_in == 'both':
                x2d, x3d = x
                p2d = F.max_pool3d(x2d, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                if x3d.shape[2] >= self.min_z:
                    p3d = F.max_pool3d(x3d, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    p3d = F.max_pool3d(x3d, kernel_size=(1, 2, 2), stride=(1, 2, 2))

                x = FMU(p2d, p3d, mode=self.FMU)

            elif self.mode_in == '2d':
                x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

            elif self.mode_in == '3d':
                if x.shape[2] >= self.min_z:
                    x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        if self.mode_out == '2d':
            return self.CB2d(x)
        elif self.mode_out == '3d':
            return self.CB3d(x)
        elif self.mode_out == 'both':
            return self.CB2d(x), self.CB3d(x)


class CAMNet(BasicNet):
    def __init__(self, in_channels, num_classes, kn=(32, 48, 64, 80, 96), ds=True, FMU='sub'):
        """
        Context-aware Mesh Network
        Args:
            in_channels: channels of input
            num_classes: output classes
            kn: the number of kernels
            ds: deep supervision
            FMU: type of feature merging unit
        """
        super().__init__()
        self.ds = ds
        self.num_classes = num_classes

        channel_factor = {'sum': 1, 'sub': 1, 'cat': 2}
        fct = channel_factor[FMU]

        self.down11 = Down(in_channels, kn[0], ('/', 'both'), downsample=False)

        self.down12 = ContextDown(kn[0], kn[1], ('2d', 'both'))
        self.down13 = ContextDown(kn[1], kn[2], ('2d', 'both'))
        self.down14 = ContextDown(kn[2], kn[3], ('2d', 'both'))
        self.bottleneck1 = ContextDown(kn[3], kn[4], ('2d', '2d'))
        self.up11 = ContextUp(fct * (kn[3] + kn[4]), kn[3], up_channels=(kn[3], kn[4]), mode=('both', '2d'))
        self.up12 = ContextUp(fct * (kn[2] + kn[3]), kn[2], up_channels=(kn[2], kn[3]), mode=('both', '2d'))
        self.up13 = ContextUp(fct * (kn[1] + kn[2]), kn[1], up_channels=(kn[1], kn[2]), mode=('both', '2d'))
        self.up14 = ContextUp(fct * (kn[0] + kn[1]), kn[0], up_channels=(kn[0], kn[1]), mode=('both', 'both'))

        self.down21 = ContextDown(kn[0], kn[1], ('3d', 'both'))
        self.down22 = ContextDown(fct * kn[1], kn[2], ('both', 'both'), FMU)
        self.down23 = ContextDown(fct * kn[2], kn[3], ('both', 'both'), FMU)
        self.bottleneck2 = ContextDown(fct * kn[3], kn[4], ('both', 'both'), FMU)
        self.up21 = ContextUp(fct * (kn[3] + kn[4]), kn[3], up_channels=(kn[3], kn[4]), mode=('both', 'both'))
        self.up22 = ContextUp(fct * (kn[2] + kn[3]), kn[2], up_channels=(kn[2], kn[3]), mode=('both', 'both'))
        self.up23 = ContextUp(fct * (kn[1] + kn[2]), kn[1], up_channels=(kn[1], kn[2]), mode=('both', '3d'))

        self.down31 = ContextDown(kn[1], kn[2], ('3d', 'both'))
        self.down32 = ContextDown(fct * kn[2], kn[3], ('both', 'both'), FMU)
        self.bottleneck3 = ContextDown(fct * kn[3], kn[4], ('both', 'both'), FMU)
        self.up31 = ContextUp(fct * (kn[3] + kn[4]), kn[3], up_channels=(kn[3], kn[4]), mode=('both', 'both'))
        self.up32 = ContextUp(fct * (kn[2] + kn[3]), kn[2], up_channels=(kn[2], kn[3]), mode=('both', '3d'))

        self.down41 = ContextDown(kn[2], kn[3], ('3d', 'both'), FMU)
        self.bottleneck4 = ContextDown(fct * kn[3], kn[4], ('both', 'both'), FMU)
        self.up41 = ContextUp(fct * (kn[3] + kn[4]), kn[3], up_channels=(kn[3], kn[4]), mode=('both', '3d'))

        self.bottleneck5 = ContextDown(kn[3], kn[4], ('3d', '3d'))

        self.outputs = nn.ModuleList(
            [nn.Conv3d(c, num_classes, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)
             for c in [kn[0], kn[1], kn[1], kn[2], kn[2], kn[3], kn[3]]]
        )

    def forward(self, x):

        """
        Down
        if self.downsample: # default true
            if self.mode_in == 'both':
                # split
                x2d, x3d = x
                p2d = F.max_pool3d(x2d, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                if x3d.shape[2] >= self.min_z:
                    p3d = F.max_pool3d(x3d, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    p3d = F.max_pool3d(x3d, kernel_size=(1, 2, 2), stride=(1, 2, 2))

                x = FMU(p2d, p3d, mode=self.FMU)

            elif self.mode_in == '2d':
                x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

            elif self.mode_in == '3d':
                if x.shape[2] >= self.min_z:
                    x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                pass

        if self.mode_out == '2d':
            return self.CB2d(x)
        elif self.mode_out == '3d':
            return self.CB3d(x)
        elif self.mode_out == 'both':
            return self.CB2d(x), self.CB3d(x)

        :param x:
        :return:
        """


        # down11=Down(in_channels, kn[0], ('/', 'both'), downsample=False)
        # X -> X_2D, X_3D
        down11 = self.down11(x)

        # self.down12 = Down(kn[0], kn[1], ('2d', 'both'))
        # X_2D -> X_2D, X_3D
        down12 = self.down12(down11[0])
        # self.down13 = Down(kn[1], kn[2], ('2d', 'both'))
        # X_2D -> X_2D, X_3D
        down13 = self.down13(down12[0])
        # self.down14 = Down(kn[2], kn[3], ('2d', 'both'))
        # X_2D -> X_2D, X_3D
        down14 = self.down14(down13[0])
        # self.bottleneck1 = Down(kn[3], kn[4], ('2d', '2d'))
        #  X_2D -> X_2D
        bottleNeck1 = self.bottleneck1(down14[0])

        # self.down21 = Down(kn[0], kn[1], ('3d', 'both'))
        # X_3D -> X_2D, X_3D
        down21 = self.down21(down11[1])
        # self.down22 = Down(fct * kn[1], kn[2], ('both', 'both'), FMU)
        # X_2D, X_3D
        down22 = self.down22([down21[0], down12[1]])
        # self.down23 = Down(fct * kn[2], kn[3], ('both', 'both'), FMU)
        down23 = self.down23([down22[0], down13[1]])
        # self.bottleneck2 = Down(fct * kn[3], kn[4], ('both', 'both'), FMU)
        bottleNeck2 = self.bottleneck2([down23[0], down14[1]])

        # self.down31 = Down(kn[1], kn[2], ('3d', 'both'))
        down31 = self.down31(down21[1])
        # self.down32 = Down(fct * kn[2], kn[3], ('both', 'both'), FMU)
        down32 = self.down32([down31[0], down22[1]])
        # self.bottleneck3 = Down(fct * kn[3], kn[4], ('both', 'both'), FMU)
        bottleNeck3 = self.bottleneck3([down32[0], down23[1]])

        # self.down41 = Down(kn[2], kn[3], ('3d', 'both'), FMU)
        down41 = self.down41(down31[1])
        # self.bottleneck4 = Down(fct * kn[3], kn[4], ('both', 'both'), FMU)
        bottleNeck4 = self.bottleneck4([down41[0], down32[1]])

        # Down(kn[3], kn[4], ('3d', '3d'))
        bottleNeck5 = self.bottleneck5(down41[1])

        # Up(fct * (kn[3] + kn[4]), kn[3], ('both', '3d'), FMU)
        # x2d: bottleNeck4[0] 小尺寸, xskip2d : down41[0] 大尺寸, x3d: bottleNeck5, xskip3d
        up41 = self.up41([bottleNeck4[0], down41[0], bottleNeck5, down41[1]])

        up31 = self.up31([bottleNeck3[0], down32[0], bottleNeck4[1], down32[1]])
        up32 = self.up32([up31[0], down31[0], up41, down31[1]])

        up21 = self.up21([bottleNeck2[0], down23[0], bottleNeck3[1], down23[1]])
        up22 = self.up22([up21[0], down22[0], up31[1], down22[1]])
        up23 = self.up23([up22[0], down21[0], up32, down21[1]])

        up11 = self.up11([bottleNeck1, down14[0], bottleNeck2[1], down14[1]])
        up12 = self.up12([up11, down13[0], up21[1], down13[1]])
        up13 = self.up13([up12, down12[0], up22[1], down12[1]])
        up14 = self.up14([up13, down11[0], up23, down11[1]])

        if self.ds:
            features = [up14[0] + up14[1], up23, up13, up32, up12, up41, up11]
            return [self.outputs[i](features[i]) for i in range(7)]
        else:
            return self.outputs[0](up14[0] + up14[1])

from lib.model.layers.utils import count_param

if __name__ == '__main__':
    n_modal = 1
    n_classes = 3

    net = CAMNet(in_channels=n_modal, num_classes=n_classes,  kn=(32, 48, 64, 80, 96), ds=False, FMU='sub')

    print(net)
    param = count_param(net)
    print('net totoal parameters: %.2fM (%d)' % (param / 1e6, param))
