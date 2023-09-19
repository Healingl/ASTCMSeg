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

from tqdm import tqdm




if __name__ == "__main__":
    yaml_config = Parser('./config/data_config/data_settings_splitting.yaml')
    save_csv_dir = yaml_config['split_dir']


    # 2D slice all prep data
    all_mr_prep_volume_data_csv = pd.read_csv(yaml_config['all_mr_prep_volume_data_csv_path'])
    all_ct_prep_volume_data_csv = pd.read_csv(yaml_config['all_ct_prep_volume_data_csv_path'])


    # ct_split: train test
    origin_ct_train_df = pd.read_csv(yaml_config['ct_train_npz_data_csv_path'])
    origin_ct_test_df = pd.read_csv(yaml_config['ct_test_npz_data_csv_path'])
    
    # 
    ct_train_patient_id_df = origin_ct_train_df[['patient_id']]
    ct_test_patient_id_df = origin_ct_test_df[['patient_id']]

    # 2D volume
    ct_train_prep_2D_volume_csv = pd.merge(all_ct_prep_volume_data_csv,
                                                   ct_train_patient_id_df, how='inner',
                                                   on=['patient_id'])
    ct_test_prep_2D_volume_csv = pd.merge(all_ct_prep_volume_data_csv,
                                       ct_test_patient_id_df, how='inner', on=['patient_id'])

    # to csv
    prep_2D_volume_ct_train_df_path = os.path.join(save_csv_dir, 'ct_train_prep_volume.csv')
    prep_2D_volume_ct_test_df_path = os.path.join(save_csv_dir, 'ct_test_prep_volume.csv')

    ct_train_prep_2D_volume_csv.to_csv(prep_2D_volume_ct_train_df_path, index=False)
    ct_test_prep_2D_volume_csv.to_csv(prep_2D_volume_ct_test_df_path, index=False)
    print('ct train test',len(ct_train_prep_2D_volume_csv),len(ct_test_prep_2D_volume_csv))
    print('ct volume feature splitting!')

    # mr_split: train test
    origin_mr_train_df = pd.read_csv(yaml_config['mr_train_npz_data_csv_path'])
    origin_mr_test_df = pd.read_csv(yaml_config['mr_test_npz_data_csv_path'])

    #
    mr_train_patient_id_df = origin_mr_train_df[['patient_id']]
    mr_test_patient_id_df = origin_mr_test_df[['patient_id']]

    # 2D volume
    mr_train_prep_2D_volume_csv = pd.merge(all_mr_prep_volume_data_csv,
                                           mr_train_patient_id_df, how='inner',
                                           on=['patient_id'])
    mr_test_prep_2D_volume_csv = pd.merge(all_mr_prep_volume_data_csv,
                                          mr_test_patient_id_df, how='inner', on=['patient_id'])

    # to csv
    prep_2D_volume_mr_train_df_path = os.path.join(save_csv_dir, 'mr_train_prep_volume.csv')
    prep_2D_volume_mr_test_df_path = os.path.join(save_csv_dir, 'mr_test_prep_volume.csv')

    mr_train_prep_2D_volume_csv.to_csv(prep_2D_volume_mr_train_df_path, index=False)
    mr_test_prep_2D_volume_csv.to_csv(prep_2D_volume_mr_test_df_path, index=False)
    print('mr train test', len(mr_train_prep_2D_volume_csv), len(mr_test_prep_2D_volume_csv))
    print('mr volume feature splitting!')



