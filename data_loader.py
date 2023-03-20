import PIL.Image as pil_image
import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
import torch
from random import randint

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from ct_utils import ctload

EASY_MODE = False

def get_relevant_scans_dir(base_dir, covid_positive=False):
    dir_names = []
    dir_names_normal = []

    for root, dirnames, filenames in os.walk(base_dir):
        for filename in filenames:
            full_path = os.path.dirname(os.path.join(root, filename))
            dir_names.append(full_path)
            if ('normal' in full_path) ^ covid_positive:
                dir_names_normal.append(full_path)
                
    img_per_scan_counter = Counter(dir_names_normal)
    wanted_scans_dirs = [scan_dir for scan_dir in img_per_scan_counter if img_per_scan_counter[scan_dir] == 35]
    return wanted_scans_dirs

def load_random_slices(scan_dir, slices_count):
    slices_paths = sorted(os.listdir(scan_dir))
    length = len(slices_paths)
    if length < slices_count:
        raise ValueError(f"count ({slices_count}) larger than number of files in dir ({length})") 

    if EASY_MODE:
        ind=10
    else:
        ind = randint(10, length - 10 - slices_count)
        
    chosen_slices_paths =  slices_paths[ind:ind + slices_count]
    return np.array([ctload(os.path.join(scan_dir, slice_path)) for slice_path in chosen_slices_paths])

def generate_mask(img_dims, mask_dims, num_masked_slices, padding_slices):
    mask = torch.zeros((num_masked_slices + padding_slices * 2, img_dims[0], img_dims[1]))
    
    if EASY_MODE:
        x_ind = y_ind = 300
    else:
        x_ind = randint(img_dims[0] // 4, img_dims[0] - (img_dims[0] // 4) - mask_dims[0])
        y_ind = randint(img_dims[1] // 4, img_dims[1] - (img_dims[1] // 4) - mask_dims[1])
        
    mask[padding_slices: -padding_slices,x_ind:x_ind + mask_dims[0],y_ind:y_ind + mask_dims[1]] = 1
    return torch.gt(mask, torch.zeros_like(mask))


class CTDataset(Dataset):
  def __init__(self, dir_path, mask_dims=(64, 64), num_masked_slices=3, padding_slices=2, transform=transforms.ToTensor(), covid_positive=False, limit_dataset=None, transforms_per_scan=10):
    self._scans_paths = sorted(get_relevant_scans_dir(dir_path, covid_positive=covid_positive))
    self._mask_dims = mask_dims
    self._num_masked_slices = num_masked_slices
    self._padding_slices = padding_slices
    if limit_dataset:
        self._scans_paths = self._scans_paths[:limit_dataset]
    self._scans_count = len(self._scans_paths)
    self._dataset_size = self._scans_count * transforms_per_scan
    # self._transform = transform
    # Cuurently ignore  transform and noraml_only

  def __len__(self):
    """
    Returns:
      the length of the dataset. 
    """
    return self._dataset_size


  def __getitem__(self, idx):
    raw_scan = load_random_slices(self._scans_paths[idx % self._scans_count], self._num_masked_slices + self._padding_slices * 2)
    
    if EASY_MODE:
        proccessed_scans = []
        for i in range(len(raw_scan)):
            resized = cv2.resize(raw_scan[i], (64, 64))
            normalized = np.where(resized < 500, 0, 1)
            proccessed_scans.append(normalized)
        raw_scan = np.array(proccessed_scans)

    if False:
        proccessed_scans = []
        for i in range(len(raw_scan)):
            resized = cv2.resize(raw_scan[i], (64, 64))
            proccessed_scans.append(resized)
        raw_scan = np.array(proccessed_scans)
        
        
    scan = torch.Tensor(raw_scan.astype(np.int32))
    mask = generate_mask(scan.shape[1:], self._mask_dims, self._num_masked_slices, self._padding_slices)
    masked = scan.clone()
    masked = masked.masked_fill_(mask, 0)
    # return {"orig": scan, "mask": mask, "masked": masked}
    return scan, mask, masked
