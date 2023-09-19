#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import os
import pandas as pd
import numpy as np
from simple_parser import Parser

if __name__ == '__main__':
    yaml_config = Parser('./config/data_config/data_settings_splitting.yaml')

    csv_dir = yaml_config['csv_dir']
    slipt_csv_dir = yaml_config['split_dir']

    if not os.path.exists(csv_dir): os.makedirs(csv_dir)
    if not os.path.exists(slipt_csv_dir): os.makedirs(slipt_csv_dir)

    # ct
    # ct_train
    ct_train_npz_txt = open(yaml_config['ct_train_npz_txt_path'], mode='r')

    ct_train_npz_file_name_list = [os.path.basename(str(line).strip().split(' ')[1]) for line in ct_train_npz_txt.readlines()]
    ct_train_npz_patient_id_list = [file_name.replace('.npz', '') for file_name in ct_train_npz_file_name_list]
    ct_train_npz_file_path_list = [os.path.join(yaml_config['ct_npz_dir'], file_name) for file_name in ct_train_npz_file_name_list]

    ct_train_npz_csv = pd.DataFrame(data={'patient_id': ct_train_npz_patient_id_list,
                                          'npz_file_path':ct_train_npz_file_path_list})

    ct_train_npz_csv.to_csv(yaml_config['ct_train_npz_data_csv_path'], index=False)

    # ct_test
    ct_test_npz_txt = open(yaml_config['ct_test_npz_txt_path'], mode='r')

    ct_test_npz_file_name_list = [os.path.basename(str(line).strip().split(' ')[1]) for line in
                                   ct_test_npz_txt.readlines()]
    ct_test_npz_patient_id_list = [file_name.replace('.npz', '') for file_name in ct_test_npz_file_name_list]
    ct_test_npz_file_path_list = [os.path.join(yaml_config['ct_npz_dir'], file_name) for file_name in
                                   ct_test_npz_file_name_list]

    ct_test_npz_csv = pd.DataFrame(data={'patient_id': ct_test_npz_patient_id_list,
                                          'npz_file_path': ct_test_npz_file_path_list})

    ct_test_npz_csv.to_csv(yaml_config['ct_test_npz_data_csv_path'], index=False)

    # all ct
    all_ct_npz_data_csv = pd.concat([ct_train_npz_csv, ct_test_npz_csv])
    all_ct_npz_data_csv.to_csv(yaml_config['all_ct_npz_data_csv_path'], index=False)

    # ct
    # mr_train
    mr_train_npz_txt = open(yaml_config['mr_train_npz_txt_path'], mode='r')

    mr_train_npz_file_name_list = [os.path.basename(str(line).strip().split(' ')[1]) for line in
                                   mr_train_npz_txt.readlines()]
    mr_train_npz_patient_id_list = [file_name.replace('.npz', '') for file_name in mr_train_npz_file_name_list]
    mr_train_npz_file_path_list = [os.path.join(yaml_config['mr_npz_dir'], file_name) for file_name in
                                   mr_train_npz_file_name_list]

    mr_train_npz_csv = pd.DataFrame(data={'patient_id': mr_train_npz_patient_id_list,
                                          'npz_file_path': mr_train_npz_file_path_list})

    mr_train_npz_csv.to_csv(yaml_config['mr_train_npz_data_csv_path'], index=False)

    # mr_test
    mr_test_npz_txt = open(yaml_config['mr_test_npz_txt_path'], mode='r')

    mr_test_npz_file_name_list = [os.path.basename(str(line).strip().split(' ')[1]) for line in
                                  mr_test_npz_txt.readlines()]
    mr_test_npz_patient_id_list = [file_name.replace('.npz', '') for file_name in mr_test_npz_file_name_list]
    mr_test_npz_file_path_list = [os.path.join(yaml_config['mr_npz_dir'], file_name) for file_name in
                                  mr_test_npz_file_name_list]

    mr_test_npz_csv = pd.DataFrame(data={'patient_id': mr_test_npz_patient_id_list,
                                         'npz_file_path': mr_test_npz_file_path_list})

    mr_test_npz_csv.to_csv(yaml_config['mr_test_npz_data_csv_path'], index=False)

    # all ct
    all_mr_npz_data_csv = pd.concat([mr_train_npz_csv, mr_test_npz_csv])
    all_mr_npz_data_csv.to_csv(yaml_config['all_mr_npz_data_csv_path'], index=False)












