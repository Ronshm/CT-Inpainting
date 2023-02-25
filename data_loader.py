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

# TODO remove
import torchvision

from ct_utils import ctload

EASY_MODE = False

def get_relevant_scans_dir(base_dir):
    dir_names = []
    dir_names_normal = []

    for root, dirnames, filenames in os.walk(base_dir):
        for filename in filenames:
            full_path = os.path.dirname(os.path.join(root, filename))
            dir_names.append(full_path)
            if 'normal' in full_path:
                dir_names_normal.append(full_path)
                
    img_per_scan_counter = Counter(dir_names_normal)
    wanted_scans_dirs = [scan_dir for scan_dir in img_per_scan_counter if img_per_scan_counter[scan_dir] == 35]
    return wanted_scans_dirs

def load_random_slices(scan_dir, slices_count):
    slices_paths = sorted(os.listdir(scan_dir))
    length = len(slices_paths)
    if length < slices_count:
        raise ValueError(f"count ({slices_count}) larger than number of files in dir ({length})") 
    ind = randint(0, length - slices_count)
    # TODO remove this const
    if EASY_MODE:
        ind=10
    chosen_slices_paths =  slices_paths[ind:ind + slices_count]
    return np.array([ctload(os.path.join(scan_dir, slice_path)) for slice_path in chosen_slices_paths])

def generate_mask(img_dims, mask_dims, num_masked_slices, padding_slices):
    mask = torch.zeros((num_masked_slices + padding_slices * 2, img_dims[0], img_dims[1]))
    x_ind = randint(0, img_dims[0] - mask_dims[0])
    y_ind = randint(0, img_dims[1] - mask_dims[1])
    if EASY_MODE:
        # TODO remove this default value
        x_ind = y_ind = 20
    mask[padding_slices: -padding_slices,x_ind:x_ind + mask_dims[0],y_ind:y_ind + mask_dims[1]] = 1
    return torch.gt(mask, torch.zeros_like(mask))

from torch.utils.data import Dataset
from torchvision import transforms

class CTDataset(Dataset):
  def __init__(self, dir_path, mask_dims=(64, 64), num_masked_slices=3, padding_slices=2, transform=transforms.ToTensor(), normal_only=True, limit_dataset=None):
    self._scans_paths = sorted(get_relevant_scans_dir(dir_path))
    self._mask_dims = mask_dims
    self._num_masked_slices = num_masked_slices
    self._padding_slices = padding_slices
    if limit_dataset:
        self._scans_paths = self._scans_paths[:limit_dataset]
    # self._transform = transform
    # Cuurently ignore  transform and noraml_only

  def __len__(self):
    """
    Returns:
      the length of the dataset. 
    """
    return len(self._scans_paths)


  def __getitem__(self, idx):
    raw_scan = load_random_slices(self._scans_paths[idx], self._num_masked_slices + self._padding_slices * 2)
    
    if EASY_MODE:
        # TODO remove processing
        import cv2
        proccessed_scans = []
        for i in range(len(raw_scan)):
            resized = cv2.resize(raw_scan[i], (64, 64))
            normalized = np.where(resized < 500, 0, 1)
            proccessed_scans.append(normalized)
        raw_scan = np.array(proccessed_scans)
        
    scan = torch.Tensor(raw_scan.astype(np.int32))
    mask = generate_mask(scan.shape[1:], self._mask_dims, self._num_masked_slices, self._padding_slices)
    masked = scan.clone()
    masked = masked.masked_fill_(mask, 0)
    # return {"orig": scan, "mask": mask, "masked": masked}
    return scan, mask, masked
