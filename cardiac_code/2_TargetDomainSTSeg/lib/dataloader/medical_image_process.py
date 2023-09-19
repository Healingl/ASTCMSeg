import nibabel as nib
import numpy as np
import torch
from PIL import Image
from nibabel.processing import resample_to_output
from scipy import ndimage
import math
import torch.nn.functional as F
import SimpleITK as sitk


def get_sample_area_by_img_centre(full_vol_dim, crop_size):
    """
    :param full_vol_dim: (y,x)
    :param crop_size: (y,x)
    :return:
    """
    assert full_vol_dim[0] >= crop_size[0], "crop size y is too big"
    assert full_vol_dim[1] >= crop_size[1], "crop size z is too big"


    # np.random.seed(seed)
    # 左上角最小情况
    centre_y = (full_vol_dim[0]) // 2
    centre_x = (full_vol_dim[1]) // 2

    min_width = centre_y - crop_size[0]/2
    min_height = centre_x - crop_size[1]/2



    if min_width < 0: min_width = 0
    if min_height < 0: min_height = 0

    if min_width >= full_vol_dim[0] - crop_size[0]: min_width = full_vol_dim[0] - crop_size[0]
    if min_height >= full_vol_dim[1] - crop_size[1]: min_height = full_vol_dim[1] - crop_size[1]

    return (int(min_width), int(min_height))

def braincmda2022_cyclegan_normalization(image_narray):
    norm_array = 2*((image_narray - image_narray.min())/(image_narray.max()-image_narray.min())) - 1
    return norm_array
def braincmda2022_minmax_normalization(image_narray):
    norm_array = (image_narray - image_narray.min())/(image_narray.max()-image_narray.min())
    return norm_array
def braincmda2022_meanstd_to_onezero_normalization(img_array):
    norm_array = (img_array+1)/2
    return norm_array
def braincmda2022_onezero_to_meanstd_normalization(img_array):
    norm_array = 2*img_array - 1
    return norm_array


def itensity_normalization(image_narray, norm_type='max_min'):
    if norm_type == 'full_volume_mean':
        norm_img_narray = (image_narray - image_narray.mean()) / image_narray.std()
    elif norm_type == 'max_min':
        norm_img_narray = (image_narray - image_narray.min()) / (image_narray.max() - image_narray.min())
    elif norm_type == 'non_normal':
        norm_img_narray = image_narray
    elif norm_type == 'non_zero_normal':
        pixels = image_narray[image_narray > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (image_narray - mean) / std
        out_random = np.random.normal(0, 1, size=image_narray.shape)
        out[image_narray == 0] = out_random[image_narray == 0]
        norm_img_narray = out
    elif norm_type == 'mr_normal':
        image_narray[image_narray > 4095] = 4095
        norm_img_narray = image_narray * 2. / 4095 - 1

    else:
        assert False
    return norm_img_narray

def read_nii_as_narray(nii_file_path):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # simple itk读取nii图像
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 读取当前病人的CT图像
    patient_itk_data = sitk.ReadImage(nii_file_path)

    # 获取CT图像的相关信息，原始坐标，spacing，方向
    origin = np.array(patient_itk_data.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(patient_itk_data.GetSpacing())  # spacing of voxels in world coor. (mm)
    direction = np.array(patient_itk_data.GetDirection())

    # 获得numpy格式的volume数据
    patient_volume_narray = sitk.GetArrayFromImage(patient_itk_data)  # z, y, x
    return patient_volume_narray, spacing

def save_narray_as_nii_file(input_narray, save_nii_file_path, spacing, origin, direction):
    savedImg = sitk.GetImageFromArray(input_narray)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(spacing)

    print('Save:', save_nii_file_path)
    sitk.WriteImage(savedImg, save_nii_file_path)

def crop_cube_from_volume(origin_volume,crop_point, crop_size):
    """

    :param origin_volume: (z,y,x)
    :param crop_point: (current_crop_z, current_crop_y, current_crop_x )
    :param crop_size: (crop_z,crop_y,crop_x)
    :return:
    """
    current_crop_z, current_crop_y, current_crop_x = crop_point

    cube = origin_volume[current_crop_z:current_crop_z + crop_size[0], current_crop_y:current_crop_y + crop_size[1], current_crop_x:current_crop_x + crop_size[2]]

    return cube

def put_cube_to_volume(origin_volume, crop_point, small_cube):
    """
    :param origin_volume: (z,y,x)
    :param crop_point: (current_crop_z, current_crop_y, current_crop_x )
    :param crop_size: (crop_z,crop_y,crop_x)
    :return:
    """
    current_crop_z, current_crop_y, current_crop_x = crop_point
    crop_size = small_cube.shape
    origin_volume[current_crop_z:current_crop_z + crop_size[0], current_crop_y:current_crop_y + crop_size[1], current_crop_x:current_crop_x + crop_size[2]] = small_cube

    return origin_volume

def get_sample_area_by_centre_crop_from_volume(full_vol_dim, crop_size):
    """
    根据中心点获得左上点采样范围, z,y,x
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


    if min_slice < 0: min_slice = 0
    if min_width < 0: min_width = 0
    if min_height < 0: min_height = 0

    return (min_slice, min_width, min_height)

def normalize_intensity(img_tensor, normalization="full_volume_mean", norm_values=(0, 1, 1, 0)):
    """
    Accepts an image tensor and normalizes it
    :param normalization: choices = "max", "mean" , type=str
    """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
    elif normalization == "max":
        max_val, _ = torch.max(img_tensor)
        img_tensor = img_tensor / max_val
    elif normalization == 'brats':
        # print(norm_values)
        normalized_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]
        final_tensor = torch.where(img_tensor == 0., img_tensor, normalized_tensor)
        final_tensor = 100.0 * ((final_tensor.clone() - norm_values[3]) / (norm_values[2] - norm_values[3])) + 10.0
        x = torch.where(img_tensor == 0., img_tensor, final_tensor)
        return x

    elif normalization == 'full_volume_mean':
        img_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]

    elif normalization == 'max_min':
        img_tensor = (img_tensor - norm_values[3]) / ((norm_values[2] - norm_values[3]))

    elif normalization == 'non_normal':
        img_tensor = img_tensor
    return img_tensor


## todo percentiles

def clip_range(img_numpy):
    """
    Cut off outliers that are related to detected black in the image (the air area)
    """
    # Todo median value!
    zero_value = (img_numpy[0, 0, 0] + img_numpy[-1, 0, 0] + img_numpy[0, -1, 0] + \
                  img_numpy[0, 0, -1] + img_numpy[-1, -1, -1] + img_numpy[-1, -1, 0] \
                  + img_numpy[0, -1, -1] + img_numpy[-1, 0, -1]) / 8.0
    non_zeros_idx = np.where(img_numpy >= zero_value)
    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
    y = img_numpy[min_z:max_z, min_h:max_h, min_w:max_w]
    return y


def percentile_clip(img_numpy, min_val=0.1, max_val=99.8):
    """
    Intensity normalization based on percentile
    Clips the range based on the quarile values.
    :param min_val: should be in the range [0,100]
    :param max_val: should be in the range [0,100]
    :return: intesity normalized image
    """
    low = np.percentile(img_numpy, min_val)
    high = np.percentile(img_numpy, max_val)

    img_numpy[img_numpy < low] = low
    img_numpy[img_numpy > high] = high
    return img_numpy
