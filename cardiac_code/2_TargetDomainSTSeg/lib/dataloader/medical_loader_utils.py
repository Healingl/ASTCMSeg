from lib.dataloader import medical_image_process as img_loader
import torch
import numpy as np
import math
import torch.nn.functional as F

def back_seg_to_origin(segmentation_map, dataset="iseg2017"):
    if dataset == "iseg2017" or dataset == "iseg2019":
        # Converts 0:background / 10:CSF / 150:GM / 250:WM to 0/1/2/3.
        segmentation_map[segmentation_map == 1] = 10
        segmentation_map[segmentation_map == 2] = 150
        segmentation_map[segmentation_map == 3] = 250
        segmentation_map = segmentation_map.astype(np.uint8)
    else:
        assert False
    return segmentation_map

def fix_seg_map(segmentation_map, dataset="iseg2017"):


    if dataset == "iseg2017" or dataset == "iseg2019":
        # Converts 0:background / 10:CSF / 150:GM / 250:WM to 0/1/2/3.
        segmentation_map[segmentation_map == 10] = 1
        segmentation_map[segmentation_map == 150] = 2
        segmentation_map[segmentation_map == 250] = 3

    elif dataset == "brats2017" or dataset == "brats2018" or dataset == "brats2019" or dataset == "brats2020":
        ED = 2
        NCR = 1
        NET_NCR = 1
        ET = 3
        segmentation_map[segmentation_map == 1] = NET_NCR
        segmentation_map[segmentation_map == 2] = ED
        segmentation_map[segmentation_map == 3] = 3
        segmentation_map[segmentation_map == 4] = 3
        segmentation_map[segmentation_map >= 4] = 3
        # print('brats label',segmentation_map.min(),segmentation_map.max())
    elif dataset == "mrbrains2013_4":
        """
        The following structures are manually segmented and will be available for training:

        1.Cortical gray matter
        2.Basal ganglia
        3.White matter
        4.White matter lesions
        5.Cerebrospinal fluid in the extracerebral space
        6.Ventricles
        7.Cerebellum
        8.Brainstem

        The numbers in front of the structures indicate the labels in the ground truth image. 
        Background will be labeled as 0. When your algorithm only uses gray matter, 
        white matter and cerebrospinal fluid labels, 
        you should merge labels 1 and 2, 3 and 4, and 5 and 6 yourself.

        website in the results section. Segmented tissue should be labeled as follows:

        1.Background (everything outside the brain)
        2.Cerebrospinal fluid (including ventricles)
        3.Gray matter (cortical gray matter and basal ganglia)
        4.White matter (including white matter lesions)

        """
        CSF = 1
        GM = 2
        WM = 3

        # print(segmentation_map.max())
        segmentation_map[segmentation_map == 1] = GM
        segmentation_map[segmentation_map == 2] = GM
        segmentation_map[segmentation_map == 3] = WM
        segmentation_map[segmentation_map == 4] = WM
        segmentation_map[segmentation_map == 5] = CSF
        segmentation_map[segmentation_map == 6] = CSF
        segmentation_map[segmentation_map > 6] = 0

    elif dataset == "mrbrains2018_4":
        """
        与MRBRAINS13不同，GM=1,WM=2,CSF=3
        Algorithms that only segment gray matter,
        white matter and cerebrospinal fluid should merge labels 1 and 2, 3 and 4, and 5 and 6, 
        and label the output as either 
        !!!0 (background), 1 (gray matter), 2 (white matter) and 3 (CSF). !!!!!!
        The cerebellum and brain stem (label 7 and 8) will in that case be excluded from the evaluation. 
        The results on the merged labels will be ranked separately.
        """
        GM = 1
        WM = 2
        CSF = 3
        # print(segmentation_map.max())
        segmentation_map[segmentation_map == 1] = GM
        segmentation_map[segmentation_map == 2] = GM
        segmentation_map[segmentation_map == 3] = WM
        segmentation_map[segmentation_map == 4] = WM
        segmentation_map[segmentation_map == 5] = CSF
        segmentation_map[segmentation_map == 6] = CSF
        segmentation_map[segmentation_map > 6] = 0

    elif dataset == "mrbrains2018_9":
        """
        The objective of this challenge is to automatically segment labels 1 to 8. 
        Ground-truth labels 9 and 10, i.e. infarctions and ‘other’ lesions, 
        will be excluded in the evaluation. 
        This means that those voxels will have no effect on the scores; 
        you may label these voxels as gray matter, white matter, or any other label. 
        The background (label 0) will not be treated as an ‘object’ itself, 
        but any false-positive label (1 to 8) in the background will negatively influence the scores.
        """
        target_label = np.zeros_like(segmentation_map)
        for i in range(1, 9):
            target_label[segmentation_map == i] = i
        segmentation_map = target_label
    else:
        assert False
    return segmentation_map


def get_order_value_list(start_idx,input_volume_axis, crop_shape_axi,extraction_step_axi):
    start_idx_list = [start_idx]
    while start_idx < input_volume_axis - crop_shape_axi:
        start_idx += extraction_step_axi
        if start_idx > input_volume_axis - crop_shape_axi:
            start_idx = input_volume_axis - crop_shape_axi
            start_idx_list.append(start_idx)
            break
        start_idx_list.append(start_idx)
    return start_idx_list

def get_order_crop_list(volume_shape,crop_shape,extraction_step):
    """
    :param volume_shape: e.g.(155,240,240)
    :param crop_shape: e.g.(128,128,128)
    :param extraction_step: e.g.(128,128,128)
    :return:
    """
    assert volume_shape[0] >= crop_shape[0], "crop size is too big"
    assert volume_shape[1] >= crop_shape[1], "crop size is too big"
    assert volume_shape[2] >= crop_shape[2], "crop size is too big"
    crop_z_list = get_order_value_list(start_idx=0,input_volume_axis=volume_shape[0],crop_shape_axi=crop_shape[0],extraction_step_axi=extraction_step[0])
    crop_y_list = get_order_value_list(start_idx=0,input_volume_axis=volume_shape[1],crop_shape_axi=crop_shape[1],extraction_step_axi=extraction_step[1])
    crop_x_list =get_order_value_list(start_idx=0,input_volume_axis=volume_shape[2],crop_shape_axi=crop_shape[2],extraction_step_axi=extraction_step[2])
    crop_list = []
    for current_crop_z_value in crop_z_list:
        for current_crop_y_value in crop_y_list:
            for current_crop_x_value in crop_x_list:
                crop_list.append((current_crop_z_value,current_crop_y_value,current_crop_x_value))
    return crop_list



def get_order_crop_2D_slice_list(volume_shape,crop_shape,extraction_step):
    """
    :param volume_shape: e.g.(240,240)
    :param crop_shape: e.g.(128,128)
    :param extraction_step: e.g.(128,128)
    :return:
    """
    assert volume_shape[0] >= crop_shape[0], "crop size is too big"
    assert volume_shape[1] >= crop_shape[1], "crop size is too big"

    crop_y_list = get_order_value_list(start_idx=0,input_volume_axis=volume_shape[0],crop_shape_axi=crop_shape[0],extraction_step_axi=extraction_step[0])
    crop_x_list = get_order_value_list(start_idx=0,input_volume_axis=volume_shape[1],crop_shape_axi=crop_shape[1],extraction_step_axi=extraction_step[1])

    crop_list = []
    for current_crop_y_value in crop_y_list:
        for current_crop_x_value in crop_x_list:
            crop_list.append((current_crop_y_value,current_crop_x_value))

    return crop_list

def get_sample_area_by_centre_crop_from_volume(full_vol_dim, crop_size, data_type='source'):
    """
    根据中心点获得左上点采样范围
    :param full_vol_dim:
    :param crop_size:
    :return:
    """
    assert full_vol_dim[0] >= crop_size[0], "crop size is too big"
    assert full_vol_dim[1] >= crop_size[1], "crop size is too big"
    assert full_vol_dim[2] >= crop_size[2], "crop size is too big"

    # 左上角最小情况
    centre_z = (full_vol_dim[0]) // 2
    centre_y = (full_vol_dim[1]) // 2
    centre_x = (full_vol_dim[2]) // 2

    min_slice = centre_z - crop_size[0]//2
    min_width = centre_y - crop_size[1]//2
    min_height = centre_x - crop_size[2]//2

    if data_type == 'source':
        # -38~+2
        min_slice = min_slice - 18
    elif data_type == 'target':
        # -30~+10
        min_slice = min_slice - 10
    else:
        assert False, 'Error data type'

    if min_slice < 0: min_slice = 0
    if min_width < 0: min_width = 0
    if min_height < 0: min_height = 0

    return (min_slice, min_width, min_height)

def get_sample_point_list_between_min_max_idx(min_dix, max_idx, crop_length, sliding_step):
    min_dix_list = [min_dix]
    while min_dix < max_idx - crop_length:
        min_dix += sliding_step
        if min_dix > max_idx - crop_length:
            min_dix = max_idx - crop_length
            min_dix_list.append(min_dix)
            break
        min_dix_list.append(min_dix)
    return min_dix_list

def get_around_mask_order_sample_list(origin_volume_size, centre_crop_min_point, crop_size, sample_sliding_step=(8,8,8)):
    volume_size = origin_volume_size
    centre_crop_min_point = centre_crop_min_point
    crop_size = crop_size
    sample_sliding_step = sample_sliding_step

    sample_region_min_z = centre_crop_min_point[0] - sample_sliding_step[0]
    sample_region_min_y = centre_crop_min_point[1] - sample_sliding_step[1]
    sample_region_min_x = centre_crop_min_point[2] - sample_sliding_step[2]

    if sample_region_min_z < 0: sample_region_min_z = 0
    if sample_region_min_y < 0: sample_region_min_y = 0
    if sample_region_min_x < 0: sample_region_min_x = 0

    sample_region_max_z = centre_crop_min_point[0] + sample_sliding_step[0]
    sample_region_max_y = centre_crop_min_point[1] + sample_sliding_step[1]
    sample_region_max_x = centre_crop_min_point[2] + sample_sliding_step[2]

    if sample_region_max_z + crop_size[0] > volume_size[0]: sample_region_max_z = volume_size[0] - crop_size[0]
    if sample_region_max_y + crop_size[1] > volume_size[1]: sample_region_max_y = volume_size[1] - crop_size[1]
    if sample_region_max_x + crop_size[2] > volume_size[2]: sample_region_max_x = volume_size[2] - crop_size[2]


    centre_crop_order_z_list = sorted(list(set([sample_region_min_z, centre_crop_min_point[0], sample_region_max_z])))
    centre_crop_order_y_list = sorted(list(set([sample_region_min_y, centre_crop_min_point[1], sample_region_max_y])))
    centre_crop_order_x_list = sorted(list(set([sample_region_min_x, centre_crop_min_point[2], sample_region_max_x])))

    sample_crop_list = []
    for current_crop_z_value in centre_crop_order_z_list:
        for current_crop_y_value in centre_crop_order_y_list:
            for current_crop_x_value in centre_crop_order_x_list:
                sample_crop_list.append((current_crop_z_value, current_crop_y_value, current_crop_x_value))

    return sample_crop_list


def get_sample_area_by_centre(mask_min_idx, mask_max_idx,full_vol_dim, crop_size):
    """
    根据中心点获得左上点采样范围
    :param mask_min_idx:
    :param mask_max_idx:
    :param full_vol_dim:
    :param crop_size:
    :return:
    """
    assert full_vol_dim[0] >= crop_size[0], "crop size is too big"
    assert full_vol_dim[1] >= crop_size[1], "crop size is too big"
    assert full_vol_dim[2] >= crop_size[2], "crop size is too big"

    # np.random.seed(seed)
    # 左上角最小情况
    centre_z = (mask_min_idx[0] + mask_max_idx[0]) // 2
    centre_y = (mask_min_idx[1] + mask_max_idx[1]) // 2
    centre_x = (mask_min_idx[2] + mask_max_idx[2]) // 2

    min_slice = centre_z - crop_size[0]//2
    min_width = centre_y - crop_size[1]//2
    min_height = centre_x - crop_size[2]//2

    if min_slice < 0: min_slice = 0
    if min_width < 0: min_width = 0
    if min_height < 0:min_height = 0

    if min_slice + crop_size[0] > full_vol_dim[0]:
        min_slice = full_vol_dim[0] - crop_size[0]

    if min_width + crop_size[1] > full_vol_dim[1]:
        min_width = full_vol_dim[1] - crop_size[1]

    if min_height + crop_size[2] > full_vol_dim[2]:
        min_height = full_vol_dim[2] - crop_size[2]

    # z,y,x
    return (min_slice, min_width, min_height)

def get_sample_area_by_mask(mask_min_idx, mask_max_idx,full_vol_dim, crop_size, seed=2020):
    """
    根据mask获得左上点采样范围
    :param mask_min_idx:
    :param mask_max_idx:
    :param full_vol_dim:
    :param crop_size:
    :return:
    """
    assert full_vol_dim[0] >= crop_size[0], "crop size is too big"
    assert full_vol_dim[1] >= crop_size[1], "crop size is too big"
    assert full_vol_dim[2] >= crop_size[2], "crop size is too big"
    # np.random.seed(seed)
    # 左上角最小情况
    min_slice = mask_min_idx[0] - crop_size[0]
    min_width = mask_min_idx[1] - crop_size[1]
    min_height = mask_min_idx[2] - crop_size[2]

    if min_slice < 0: min_slice=0
    if min_width < 0: min_width=0
    if min_height < 0: min_height=0

    # 右下角最大情况,如果crop的左上角点任意坐标方向比mask区域右下角坐标都大，那么肯定不相交
    max_slice = mask_max_idx[0]
    max_width = mask_max_idx[1]
    max_height = mask_max_idx[2]

    #
    if max_slice >= full_vol_dim[0] - crop_size[0]: max_slice = full_vol_dim[0] - crop_size[0]
    if max_width >= full_vol_dim[1] - crop_size[1]: max_width = full_vol_dim[1] - crop_size[1]
    if max_height >= full_vol_dim[2] - crop_size[2]: max_height = full_vol_dim[2] - crop_size[2]

    if full_vol_dim[0] == crop_size[0]:
        slices = 0
    else:
        slices = np.random.randint(low=min_slice,high=max_slice)

    if full_vol_dim[1] == crop_size[1]:
        w_crop = 0
    else:
        w_crop = np.random.randint(low=min_width,high=max_width)

    if full_vol_dim[2] == crop_size[2]:
        h_crop = 0
    else:
        h_crop = np.random.randint(low=min_height,high=max_height)

    return (slices, w_crop, h_crop)




