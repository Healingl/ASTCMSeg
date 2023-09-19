#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 23-9-2
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import torch
from .base_model import BaseModel
from . import networks
import lib.models.util.util as util
from torch.distributions.beta import Beta
from torch.nn import functional as F
from lib.models.WNCE import PatchWNCELoss
from lib.models.ARC import ARC_Loss
import torch.nn as nn
import itertools

from lib.models.freq_translation.freq_fourier_loss import fft_L1_loss_color, decide_circle, fft_L1_loss_mask
from lib.models.freq_translation.freq_pixel_loss import find_fake_freq, get_gaussian_kernel

import matplotlib.pyplot as plt

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class ACMIT(BaseModel):
    """
    A 3D Anatomy-Guided Self-Training Segmentation Framework for Unpaired Cross-Modality Medical Image Segmentation
    Zhuang et al.
    """
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.train_epoch = None

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'Freq', 'DICE']

        if opt.lambda_WNCE > 0.0:
            self.loss_names.append('WNCE')
            if opt.nce_idt and self.isTrain:
                self.loss_names += ['WNCE_Y']

        if opt.lambda_ARC > 0.0:
            self.loss_names.append('ARC')

        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = self.opt.nce_layers
        self.alpha = opt.alpha
        if opt.nce_idt and self.isTrain:
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'S']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        #
        if opt.netSeg == 'unet':
            from lib.models.unet import LatentFeatureSegDecoder
            self.netS = LatentFeatureSegDecoder(shared_code_channel=opt.ngf * 4, out_channel=opt.output_nc_seg).cuda()

        else:
            assert False, 'No seg model'

        self.gauss_kernel = get_gaussian_kernel(opt['gauss_kernel_size']).cuda()

        img_size = 256
        batch = opt['train_batch_size']

        # (2,256,256)
        mask_h, mask_l = decide_circle(r=opt['gauss_kernel_size'], N=int(batch), L=img_size)
        mask_h = mask_h.unsqueeze_(dim=1).repeat(1, opt['input_nc'], 1, 1)
        mask_l = mask_l.unsqueeze_(dim=1).repeat(1, opt['input_nc'], 1, 1)
        self.mask_h, self.mask_l = mask_h.cuda(), mask_l.cuda()

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                          opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)

            self.criterionWNCE = []

            for i, nce_layer in enumerate(self.nce_layers):
                self.criterionWNCE.append(PatchWNCELoss(opt=opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            # seg
            from torch.nn.modules.loss import CrossEntropyLoss
            from lib.loss.dice_bce_loss import DiceLoss
            self.dice_criterion = DiceLoss().to(self.device)
            self.ce_criterion = CrossEntropyLoss().to(self.device)

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netS.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.criterionR = []
            for nce_layer in self.nce_layers:
                self.criterionR.append(ARC_Loss(opt).to(self.device))

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()  # calculate gradients for D
            self.compute_G_loss().backward()  # calculate graidents for G
            # if self.opt.lambda_NCE > 0.0:
            #     self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
            #     self.optimizers.append(self.optimizer_F)
            #
            # elif self.opt.lambda_WNCE > 0.0:
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            # if self.opt.lambda_NCE > 0.0:
            #     self.optimizer_F.zero_grad()
            # elif self.opt.lambda_WNCE > 0.0:
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            # if self.opt.lambda_NCE > 0.0:
            #     self.optimizer_F.step()
            # elif self.opt.lambda_WNCE > 0.0:
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.seg_real_A = input['Seg'].to(self.device)
        self.image_paths = 'None'

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B),
                              dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def set_epoch(self, epoch):
        self.train_epoch = epoch

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        ## get feat
        fake_B_feat = self.netG(self.fake_B, self.nce_layers, encode_only=True)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            fake_B_feat = [torch.flip(fq, [3]) for fq in fake_B_feat]

        real_A_feat = self.netG(self.real_A, self.nce_layers, encode_only=True)

        t1_bottleneck_feature = real_A_feat[-1]


        fake_B_pool, sample_ids = self.netF(fake_B_feat, self.opt.num_patches, None)
        real_A_pool, _ = self.netF(real_A_feat, self.opt.num_patches, sample_ids)

        if self.opt.nce_idt:
            idt_B_feat = self.netG(self.idt_B, self.nce_layers, encode_only=True)
            if self.opt.flip_equivariance and self.flipped_for_equivariance:
                idt_B_feat = [torch.flip(fq, [3]) for fq in idt_B_feat]
            real_B_feat = self.netG(self.real_B, self.nce_layers, encode_only=True)

            idt_B_pool, _ = self.netF(idt_B_feat, self.opt.num_patches, sample_ids)
            real_B_pool, _ = self.netF(real_B_feat, self.opt.num_patches, sample_ids)

        # different point

        ## Relation Loss
        self.loss_ARC, weight = self.calculate_R_loss(real_A_pool, fake_B_pool, epoch=self.train_epoch)

        ## WNCE
        if self.opt.lambda_WNCE > 0.0:
            self.loss_WNCE = self.calculate_WNCE_loss(real_A_pool, fake_B_pool, weight)
        else:
            self.loss_WNCE, self.loss_WNCE_bd = 0.0, 0.0

        self.loss_WNCE_Y = 0
        if self.opt.nce_idt and self.opt.lambda_WNCE > 0.0:
            _, weight_idt = self.calculate_R_loss(real_B_pool, idt_B_pool, only_weight=True, epoch=self.train_epoch)
            self.loss_WNCE_Y = self.calculate_WNCE_loss(real_B_pool, idt_B_pool, weight_idt)
            loss_WNCE_both = (self.loss_WNCE + self.loss_WNCE_Y) * 0.5
        else:
            loss_WNCE_both = self.loss_WNCE

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # Calculate Frequency Domain Loss
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # High Frequency Consistency
        fft_swap_H = fft_L1_loss_mask(self.fake_B, self.real_A, self.mask_h)
        self.loss_Freq = fft_swap_H

        # Reconstruct Frequency Consistency
        if self.opt.nce_idt:
            # x_rec, x_real
            loss_recon_fft = fft_L1_loss_color(self.idt_B, self.real_B)
            self.loss_Freq += loss_recon_fft

        self.loss_Freq = self.loss_Freq * self.opt.lambda_Fre

        if self.opt.lambda_DICE > 0.0:
            self.seg_real_t1 = self.netS(t1_bottleneck_feature)
            loss_seg_real_t1 = self.dice_criterion(self.seg_real_t1, self.seg_real_A) + self.ce_criterion(
                self.seg_real_t1, self.seg_real_A)
            self.loss_DICE = (loss_seg_real_t1) * self.opt.lambda_DICE
        else:
            self.loss_DICE = 0.0

        self.loss_G = self.loss_G_GAN + loss_WNCE_both + self.loss_ARC + self.loss_Freq + self.loss_DICE

        return self.loss_G

    def calculate_WNCE_loss(self, src, tgt, weight=None):
        n_layers = len(self.nce_layers)

        feat_q_pool = tgt
        feat_k_pool = src

        total_WNCE_loss = 0.0
        for f_q, f_k, crit, nce_layer, w in zip(feat_q_pool, feat_k_pool, self.criterionWNCE, self.nce_layers, weight):
            if self.opt.no_Hneg:
                w = None
            loss = crit(f_q, f_k, w) * self.opt.lambda_WNCE
            total_WNCE_loss += loss.mean()

        return total_WNCE_loss / n_layers

    def calculate_R_loss(self, src, tgt, only_weight=False, epoch=None):
        n_layers = len(self.nce_layers)

        feat_q_pool = tgt
        feat_k_pool = src

        total_ARC_loss = 0.0
        weights = []
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionR, self.nce_layers):
            # print('f_q.shape, f_k.shape',f_q.shape, f_k.shape)
            loss_ARC, weight = crit(f_q, f_k, only_weight, epoch)
            total_ARC_loss += loss_ARC * self.opt.lambda_ARC
            weights.append(weight)
        return total_ARC_loss / n_layers, weights

# --------------------------------------------------------------------------------------------------------
