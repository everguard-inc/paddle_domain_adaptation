import os
from typing import List, NoReturn, Optional, Any, Dict
import yaml
import cv2
import jpeg4py as jpeg
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augmentation(mode='train', input_width=1280, input_height=736):
    if mode == 'train':
        transform = [
            A.Resize(input_height, input_width),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
            A.HueSaturationValue(sat_shift_limit=25, hue_shift_limit=25, val_shift_limit=25, p=0.5),
            A.ImageCompression(quality_lower=90, p=0.5),
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=0.05, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            #ToTensorV2(),
        ]
    elif mode == 'val': 
        transform = [
            A.Resize(input_height, input_width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            #ToTensorV2(),
        ]
    return A.Compose(transform, additional_targets={'edge_mask': 'mask'})