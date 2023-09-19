# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform


class ResConvBlock(nn.Module):
    """ResConv"""

    def __init__(self, in_channels):
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),

        )
    def forward(self, x):
        return self.conv2(x+self.conv1(x))




class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, stride=1, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output



def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x

class UNet(nn.Module):
    def __init__(self, in_chns, class_num, base_filter=16):
        super(UNet, self).__init__()

        params = {
                  'in_chns': in_chns,
                  'feature_chns': [base_filter, base_filter*2, base_filter*4, base_filter*8, base_filter*16],
                  'dropout': [0.0, 0.0, 0.0, 0.0, 0.0],
                  'class_num': class_num,
                  'bilinear': False,
        }

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output

class UNetEncoder(nn.Module):
    def __init__(self, in_chns, class_num, base_filter=16):
        super(UNetEncoder, self).__init__()

        params = {
                  'in_chns': in_chns,
                  'feature_chns': [base_filter, base_filter*2, base_filter*4, base_filter*8, base_filter*16],
                  'dropout': [0.0, 0.0, 0.0, 0.0, 0.0],
                  'class_num': class_num,
                  'bilinear': False,
        }

        self.encoder = Encoder(params)


    def forward(self, modal_input):
        feature_list = self.encoder(modal_input)
        return feature_list


class UNetDecoder(nn.Module):
    def __init__(self, in_chns, class_num, base_filter=16):
        super(UNetDecoder, self).__init__()

        params = {
            'in_chns': in_chns,
            'feature_chns': [base_filter, base_filter * 2, base_filter * 4, base_filter * 8, base_filter * 16],
            'dropout': [0.0, 0.0, 0.0, 0.0, 0.0],
            'class_num': class_num,
            'bilinear': False,
        }

        self.decoder = Decoder(params)

    def forward(self, feature_list):
        output = self.decoder(feature_list)
        return output

# [2, 256, 64, 64]
from lib.models.util.model_util import Conv2dBlock,ResBlocks
class LatentFeatureSegDecoder(nn.Module):
    def __init__(self, shared_code_channel, out_channel):
        super(LatentFeatureSegDecoder, self).__init__()
        self.shared_code_channel = shared_code_channel

        self.bottleneck_layer = nn.Sequential(Conv2dBlock(shared_code_channel, shared_code_channel, 3, stride=1, padding=1, norm='in', activation='relu', pad_type='reflect', bias=False),
                                              ResBlocks(3, shared_code_channel, 'in', 'relu', pad_type='zero')
                                              )
        self.up1 = nn.Sequential(
            # input: 1/4 * 1/4
            nn.ConvTranspose2d(shared_code_channel, shared_code_channel//2,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=True),
            nn.InstanceNorm2d(shared_code_channel//2),
            nn.ReLU(True),
            Conv2dBlock(shared_code_channel//2, shared_code_channel//2, 3, 1, 1, norm='in', activation='relu', pad_type='zero'),
        )

        self.up2 = nn.Sequential(
            # input: 1/4 * 1/4
            nn.ConvTranspose2d(shared_code_channel//2, shared_code_channel//4,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=True),
            nn.InstanceNorm2d(shared_code_channel//4),
            nn.ReLU(True),
            Conv2dBlock(shared_code_channel//4, shared_code_channel//4, 3, 1, 1, norm='in', activation='relu', pad_type='zero'),
        )

        self.outconv = nn.Conv2d(shared_code_channel//4, out_channel, 3, 1, 1)


    def forward(self, shared_code):
        output = self.bottleneck_layer(shared_code)
        output = self.up1(output)
        output = self.up2(output)
        output = self.outconv(output)
        return output
