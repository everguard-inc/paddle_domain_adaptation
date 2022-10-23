# This file is made available under Apache License, Version 2.0
# This file is based on code available under the MIT License here:
#  https://github.com/ZJULearning/MaxSquareLoss/blob/master/datasets/GTA5Dataset.py
#
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, cv2
import numpy as np
from PIL import Image, ImageFile
from datasets.domen1_dataset import Domen1Dataset

import paddle
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Domen2Dataset(Domen1Dataset):
    def __init__(self,
                 images_list,
                 images_root,
                 masks_root,
                 split='train',
                 training=True,
                 edge=False,
                 logger=None):
        super(self.__class__, self).__init__(images_list, images_root, masks_root)

        # Files
        self.images_list = images_list
        self.images_root = images_root
        self.masks_root = masks_root
        self.split = split
        self.training = training
        self.logger = logger
        self.edge = edge
        # Label map
        self.id_to_trainid = {
            1: 0
        }

    def __getitem__(self, item):
        # Open image and label
        image_path = self.images_list[item]
        mask_path = os.path.join(self.masks_root, image_path[:-3]+'png')
        image_path = os.path.join(self.images_root, image_path)
        image = Image.open(image_path).convert("RGB")
        gt_image = Image.open(mask_path)

        # Augmentation
        if (self.split == "train" or
                self.split == "trainval") and self.training:
            image, gt_image, edge_mask = self._train_sync_transform(image, gt_image)
            if len(gt_image.shape)==3:
                gt_image = gt_image[:,:,0]
            return image, gt_image, edge_mask
        else:
            image, gt_image, edge_mask = self._val_sync_transform(image, gt_image)
            if len(gt_image.shape)==3:
                gt_image = gt_image[:,:,0]
            return image, gt_image, edge_mask, id

        return image, gt_image, edge_mask
