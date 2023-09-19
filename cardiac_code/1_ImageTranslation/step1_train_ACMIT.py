#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from src.utils import Recorder, Plotter

import os
import torch
from torch.utils.data import DataLoader
from lib.utils.logging import *
from lib.utils.simple_parser import Parser
import numpy as np
import shutil
import torch.nn as nn
from lib.dataloader.CMDA2DSliceDataset import CMDA2DSliceDataset

import random
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('Agg')

from tqdm import tqdm

import time
current_time = str(time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time())))


if __name__ == "__main__":
    model_config_path = './config/model_config/ACMIT_SynModel.yaml'

    model_yaml_config = Parser(model_config_path)

    model_name = model_yaml_config['model_name']
    workdir = model_yaml_config['workdir']
    if not os.path.exists(workdir): os.makedirs(workdir)

    logger = get_logger(os.path.join(workdir, '%s.log' % (model_name)))
    shutil.copy(model_config_path, os.path.join(workdir, os.path.basename(model_config_path)))

    logger.info(str(model_yaml_config))

    gpu_list = model_yaml_config['gpus']

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Random Seed
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    np.random.seed(model_yaml_config['seed'])
    random.seed(model_yaml_config['seed'])
    torch.cuda.manual_seed_all(model_yaml_config['seed'])
    torch.random.manual_seed(model_yaml_config['seed'])
    random.seed(model_yaml_config['seed'])

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Loading Dataset
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # unpaired source T1 and target T2 dataset
    # train
    train_labeled_t1_slices_dataset = CMDA2DSliceDataset(
        sample_2d_slice_csv_path=model_yaml_config['labeled_t1_slices_csv_path'],
        mode='train',
        annotation_type='labeled',
        modality='t1',
        data_num=-1,
        use_aug=True)

    train_unlabeled_t2_slices_dataset = CMDA2DSliceDataset(
        sample_2d_slice_csv_path=model_yaml_config['unlabeled_t2_slices_csv_path'],
        mode='train',
        annotation_type='unlabeled',
        modality='t2',
        data_num=-1,
        use_aug=True)
    # val
    val_labeled_t1_slices_dataset = CMDA2DSliceDataset(
        sample_2d_slice_csv_path=model_yaml_config['labeled_t1_slices_csv_path'],
        mode='val',
        annotation_type='labeled',
        modality='t1',
        data_num=-1,
        use_aug=False)

    val_unlabeled_t2_slices_dataset = CMDA2DSliceDataset(
        sample_2d_slice_csv_path=model_yaml_config['unlabeled_t2_slices_csv_path'],
        mode='val',
        annotation_type='unlabeled',
        modality='t2',
        data_num=-1,
        use_aug=False)

    train_labeled_t1_slices_loader = DataLoader(train_labeled_t1_slices_dataset, shuffle=True,
                                                batch_size=model_yaml_config['train_batch_size'],
                                                drop_last=True,
                                                pin_memory=True,
                                                num_workers=1)
    train_unlabeled_t2_slices_loader = DataLoader(train_unlabeled_t2_slices_dataset, shuffle=True,
                                                  batch_size=model_yaml_config['train_batch_size'],
                                                  drop_last=True,
                                                  pin_memory=True,
                                                  num_workers=1)

    val_labeled_t1_slices_loader = DataLoader(val_labeled_t1_slices_dataset, shuffle=True, batch_size=model_yaml_config['eval_batch_size'],
                                              num_workers=0)
    val_unlabeled_t2_slices_loader = DataLoader(val_unlabeled_t2_slices_dataset, shuffle=True, batch_size=model_yaml_config['eval_batch_size'],
                                                num_workers=0)

    logger.info('train_labeled_t1_slices_loader: %s, train_unlabeled_t2_slices_loader: %s' % (
    len(train_labeled_t1_slices_loader), len(train_unlabeled_t2_slices_loader)))

    logger.info('val_labeled_t1_slices_loader: %s, val_unlabeled_t2_slices_loader: %s' % (
    len(val_labeled_t1_slices_loader), len(val_unlabeled_t2_slices_loader)))

    torch.cuda.set_device('cuda:{}'.format(gpu_list[0]))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 训练开始
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


    # Setup Model
    input_size = (model_yaml_config['input_nc'], model_yaml_config['image_size'][0], model_yaml_config['image_size'][1])
    upsample = nn.Upsample(size=(input_size[1], input_size[2]), mode='bilinear', align_corners=True)

    logger.info('building models ...')
    assert model_yaml_config['output_nc_seg'] == len(model_yaml_config['cate_to_label_dict']), 'Error Class Num: %s' % (
        model_yaml_config['output_nc_seg'])

    from lib.models.ACMIT_model import ACMIT

    model = ACMIT(model_yaml_config)

    if model_yaml_config['isTrain'] and model_yaml_config['resume']:
        logger.info('Loading pretrained_path')

        netG_weight_path = os.path.join(model_yaml_config['resume_weight_dir'], 'latest_net_G.pth')
        model.netG.load_state_dict(torch.load(netG_weight_path, map_location='cpu'))
        logger.info('load: %s' % (netG_weight_path))


        netD_weight_path = os.path.join(model_yaml_config['resume_weight_dir'], 'latest_net_D.pth')
        model.netD.load_state_dict(torch.load(netD_weight_path, map_location='cpu'))
        logger.info('load: %s' % (netD_weight_path))

        netS_weight_path = os.path.join(model_yaml_config['resume_weight_dir'], 'latest_net_S.pth')
        model.netS.load_state_dict(torch.load(netS_weight_path, map_location='cpu'))
        logger.info('load: %s' % (netS_weight_path))

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # # Training
    # # # # # # # # # # # # # # # # # # # # # # # # #

    # all_iterations = model_yaml_config['each_epoch_iter'] * model_yaml_config['num_epochs']

    current_epoch = 0
    best_loss = 0
    best_iter = 0

    true_label = 1
    fake_label = 0

    recorder = Recorder(model.loss_names)
    plotter = Plotter(workdir, keys1=model.loss_names)

    for current_epoch in range(model_yaml_config['n_epochs']+model_yaml_config['n_epochs_decay']):
        logger.info('[Epoch %s/%s]' % (current_epoch, model_yaml_config['n_epochs']+model_yaml_config['n_epochs_decay']))
        model.set_epoch(epoch=current_epoch)

        train_labeled_t1_slices_loader_iter = iter(train_labeled_t1_slices_loader)
        train_unlabeled_t2_slices_loader_iter = iter(train_unlabeled_t2_slices_loader)

        epoch_all_iter_num = min(len(train_labeled_t1_slices_loader),len(train_unlabeled_t2_slices_loader))

        loss_record = {}
        for loss_name in model.loss_names:
            loss_record[loss_name] = []


        for current_epoch_iter in tqdm(range(epoch_all_iter_num), total=epoch_all_iter_num):
            # if current_epoch_iter > 20:
            #     break

            # # # # # # # # # # # # # # # # # # # # # # # # #
            # # Getting tensor
            # # # # # # # # # # # # # # # # # # # # # # # # #

            # source t1
            labeled_t1_feature_tensor, labeled_t1_seg_gt_tensor = next(train_labeled_t1_slices_loader_iter)

            try:
                # target t2
                unlabeled_t2_feature_tensor, _ = next(train_unlabeled_t2_slices_loader_iter)
            except StopIteration:
                train_unlabeled_t2_slices_loader_iter = iter(train_unlabeled_t2_slices_loader)
                unlabeled_t2_feature_tensor, _ = next(train_unlabeled_t2_slices_loader_iter)

            # Set model input
            labeled_t1_feature_tensor, labeled_t1_seg_gt_tensor = labeled_t1_feature_tensor.cuda(non_blocking=True), labeled_t1_seg_gt_tensor.cuda(non_blocking=True)
            unlabeled_t2_feature_tensor = unlabeled_t2_feature_tensor.cuda(non_blocking=True)
            # print(labeled_t1_feature_tensor.shape, unlabeled_t2_feature_tensor.shape)

            current_input_data = {'A': labeled_t1_feature_tensor,
                                  'Seg': labeled_t1_seg_gt_tensor,
                                  'B': unlabeled_t2_feature_tensor,
                                  'A_paths': 'None CUT A path',
                                  'B_paths': 'None CUT A path'}

            if current_epoch == 0 and current_epoch_iter == 0:
                model.data_dependent_initialize(data=current_input_data)
                model.setup(model_yaml_config)  # regular setup: load and print networks; create schedulers

            model.set_input(current_input_data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            errors = model.get_current_losses()

            loss_log = ""
            for k, v in errors.items():
                loss_log += '[%s: %.4f] ' % (k, v)
                loss_record[k].append(v)

            current_epoch_loss = loss_log
            update_log = loss_log
            current_lr = model.optimizers[0].param_groups[0]['lr']

            logger.info('[Epoch %d / %d], [iter %d / %d], [lr: %.6f], %s' % (current_epoch,
                                                                             model_yaml_config['n_epochs'] +
                                                                             model_yaml_config['n_epochs_decay'],
                                                                             epoch_all_iter_num,
                                                                             current_epoch_iter,
                                                                             current_lr,
                                                                             update_log ))

        # Update learning rates

        lr_changing_log = model.update_learning_rate()
        logger.info('###' * 20)
        logger.info(lr_changing_log)
        logger.info('###' * 20)


        # Eval
        from torchvision.utils import save_image, make_grid

        eval_img_dir = os.path.join(workdir, 'eval_syn_imgs/')
        if not os.path.exists(eval_img_dir): os.mkdir(eval_img_dir)

        logger.info('>>>>' * 30)
        logger.info('Eval Synthesis Image:')

        G_AB = model.netG
        G_AB.eval()
        with torch.no_grad():

            # source t1
            val_labeled_t1_feature_tensor, labeled_t1_seg_gt_tensor = next(iter(val_labeled_t1_slices_loader))

            # target t2
            val_unlabeled_t2_feature_tensor, _ = next(iter(val_unlabeled_t2_slices_loader))


            real_A = val_labeled_t1_feature_tensor.cuda()
            fake_B = G_AB(real_A)
            real_B = val_unlabeled_t2_feature_tensor.cuda()



        def revert_intensity(data):
            return (data+1)/2


        real_A = revert_intensity(real_A)
        fake_B = revert_intensity(fake_B)
        real_B = revert_intensity(real_B)

        real_A = make_grid(real_A, nrow=model_yaml_config['eval_batch_size'], normalize=True)
        fake_B = make_grid(fake_B, nrow=model_yaml_config['eval_batch_size'], normalize=True)
        real_B = make_grid(real_B, nrow=model_yaml_config['eval_batch_size'], normalize=True)

        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B), 1)

        current_iter_eval_img_path = os.path.join(eval_img_dir, 'epoch_%s_iter_%s_syn_img.png'%(current_epoch,epoch_all_iter_num))
        save_image(image_grid, current_iter_eval_img_path, normalize=False)

        logger.info('>>>>' * 30)

        recorder_dict = {}
        for key, value in loss_record.items():
            recorder_dict['%s' % (key)] = np.mean(value)

        recorder.update(recorder_dict)
        plotter.send(recorder.call())

        logger.info('Epoch loss: %s'%(str(recorder_dict)))

        if current_epoch % 100 == 0 and current_epoch != 0:
            model.save_networks(current_epoch)

        model.save_networks('latest')
        logger.info('>>>>' * 30)

        G_AB.train()


