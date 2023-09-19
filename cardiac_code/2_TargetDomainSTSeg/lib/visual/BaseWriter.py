import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import lib.utils as utils

dict_class_names = {"iseg2017": ["Background", "CSF", "GM", "WM"],
                    "iseg2019": ["Background", "CSF", "GM", "WM"],

                    "brainptm": ['bg','cst_left_seg','cst_right_seg','or_left_seg','or_right_seg'],

                    "brainptm_cst_left": ['bg','cst_left_seg'],

                    "brainptm_cst_right": ['bg', 'cst_right_seg'],

                    "brainptm_or_left": ['bg', 'or_left_seg'],

                    "brainptm_or_right": ['bg', 'or_right_seg'],

                    "mrbrains2013_4": ["Background", "CSF", "GM", "WM"],
                    "mrbrains2018_4": ["Background", "GM", "WM", "CSF"],

                    "mrbrains2018_9": ["Background", "Cort.GM", "BS", "WM", "WML", "CSF",
                                  "Ventr.", "Cerebellum", "stem"],

                    "brats2017": ["0Background", "1NCR/NET", "2ED", "3ET"],
                    "brats2018": ["Background", "NCR/NET", "ED", "ET"],
                    "brats2019": ["Background", "NCR", "ED", "NET", "ET"],
                    "brats2020": ["Background", "NCR/NET", "ED", "ET"],
                    }


class TensorboardWriter():

    def __init__(self, args, current_time):

        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.name_model = '%s_%s_%s'%(self.model_name,self.dataset_name,current_time)
        self.current_log_dir = os.path.join(args.logdir,self.name_model)
        if not os.path.exists(self.current_log_dir):os.makedirs(self.current_log_dir)
        print("Tensorboard log_dir = {}".format(self.current_log_dir))
        self.writer = SummaryWriter(log_dir=self.current_log_dir, comment=self.name_model)

        if self.dataset_name not in dict_class_names:
            assert False

        self.label_names = dict_class_names[self.dataset_name]
        self.data = self.create_data_structure()

    def create_data_structure(self, ):
        data = {"train": dict((label, 0.0) for label in self.label_names),
                "val": dict((label, 0.0) for label in self.label_names)}
        data['train']['loss'] = 0.0
        data['val']['loss'] = 0.0
        data['train']['count'] = 1.0
        data['val']['count'] = 1.0
        data['train']['dsc'] = 0.0
        data['val']['dsc'] = 0.0
        return data

    def reset(self, mode):
        self.data[mode]['dsc'] = 0.0
        self.data[mode]['loss'] = 0.0
        self.data[mode]['count'] = 1
        for i in range(len(self.label_names)):
            self.data[mode][self.label_names[i]] = 0.0


    def update_scores(self, iter, loss, channel_score, mode, writer_step, lr=0.0):
        """
        :param iter: iteration or partial epoch
        :param loss: any loss torch.tensor.item()
        :param channel_score: per channel score or dice coef
        :param mode: train or val ( for training and validation)
        :param writer_step: tensorboard writer step
        """
        #  [mean dice %.5f],
        updata_log = ''
        # WARNING ASSUMING THAT CHANNELS IN SAME ORDER AS DICTIONARY

        # 平均获得dice score, 百分制表示dsc
        # dice_coeff = np.mean(channel_score) * 100

        dice_coeff = np.mean(channel_score)
        num_channels = len(channel_score)

        # loss
        updata_log += ' [loss %.5f], '%(loss)
        # dice
        updata_log += ' [mean dice %.5f], ' % (dice_coeff)

        self.data[mode]['dsc'] += dice_coeff
        self.data[mode]['loss'] += loss
        self.data[mode]['count'] = iter + 1

        if mode == 'train':
            self.writer.add_scalar(mode + '/' + 'LearningRate', lr, global_step=writer_step)
            updata_log += ' [lr %.4f], ' % (lr)

        for i in range(num_channels):
            self.data[mode][self.label_names[i]] += channel_score[i]
            if self.writer is not None:
                updata_log += ' [%s %.4f], ' % (self.label_names[i],channel_score[i])
                # train/
                self.writer.add_scalar(mode + '/' + self.label_names[i], channel_score[i], global_step=writer_step)

        return updata_log

    def display_epoch_metrics(self, iter, epoch, mode='train'):
        """

        :param iter: iteration or partial epoch
        :param epoch: epoch of training
        :param loss: any loss numpy
        :param mode: train or val ( for training and validation)
        """
        display_epoch_metrics_log = '[mode:%s], [epoch %d], [loss: %.4f], [dsc: %.4f],'%((mode, epoch,
                                                                                     self.data[mode]['loss'] /
                                                                                     self.data[mode]['count'],
                                                                                     self.data[mode]['dsc'] /
                                                                                     self.data[mode]['count']))
        # [epoch %d / %d]


        for i in range(len(self.label_names)):
            display_epoch_metrics_log += ' [%s %.4f], ' % (self.label_names[i], self.data[mode][self.label_names[i]] / self.data[mode]['count'])

        return display_epoch_metrics_log

    def write_end_of_epoch(self, epoch, ndigits=4):
        self.writer.add_scalars('DSC/', {'train': self.data['train']['dsc'] / self.data['train']['count'],
                                         'val': self.data['val']['dsc'] / self.data['val']['count'],
                                         }, epoch)
        self.writer.add_scalars('Loss/', {'train': self.data['train']['loss'] / self.data['train']['count'],
                                          'val': self.data['val']['loss'] / self.data['val']['count'],
                                          }, epoch)
        for i in range(len(self.label_names)):
            self.writer.add_scalars(self.label_names[i],
                                    {'train': self.data['train'][self.label_names[i]] / self.data['train']['count'],
                                     'val': self.data['val'][self.label_names[i]] / self.data['train']['count'],
                                     }, epoch)

        end_of_epoch_dict = {'epoch':epoch,
                             'train_loss':round(self.data['train']['loss'] / self.data['train']['count'], ndigits),
                             'train_dsc': round(self.data['train']['dsc'] / self.data['train']['count'], ndigits),
                             'val_loss': round(self.data['val']['loss'] / self.data['val']['count'], ndigits),
                             'val_dsc': round(self.data['val']['dsc'] / self.data['val']['count'], ndigits)
                             }


        return end_of_epoch_dict