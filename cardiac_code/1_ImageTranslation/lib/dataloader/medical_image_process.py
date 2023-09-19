import nibabel as nib
import numpy as np
import torch
from PIL import Image
from nibabel.processing import resample_to_output
from scipy import ndimage
import math
import torch.nn.functional as F
import SimpleITK as sitk

def braincmda2022_cyclegan_normalization(image_narray):
    norm_array = 2*image_narray - 1
    return norm_array

def braincmda2022_zeroone_normalization(image_narray):
    norm_array = (image_narray + 1)/2
    return norm_array

def braincmda2022_cut_normalization(image_narray):
    norm_array = (image_narray - 0)/(255-0)
    return norm_array

def braincmda2022_minmax_normalization(image_narray):
    norm_array = (image_narray - image_narray.min())/(image_narray.max()-image_narray.min())
    return norm_array