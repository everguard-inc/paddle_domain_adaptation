# This file is made available under Apache License, Version 2.0
# This file is based on code available under the MIT License here:
#  https://github.com/ZJULearning/MaxSquareLoss/blob/master/datasets/cityscapes_Dataset.py
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
import random
import numpy as np
import collections.abc as abc
from PIL import Image, ImageOps, ImageFilter, ImageFile

import paddle
from paddle import io

import paddleseg.transforms.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_MEAN = np.array(
    (104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

label_colours = list(
    map(
        tuple,
        [
            [255, 0, 0],
        ]))

# Labels
ignore_label = 255
domen1_id_to_trainid = {
    0: ignore_label,
    1: 0
}


class Domen1Dataset(io.Dataset):
    def __init__(self,
                 images_list,
                 images_root,
                 masks_root,
                 split='train',
                 num_classes = 1,
                 training=True,
                 edge=False,
                 logger=None,
                 resize = (1280, 736)):
        self.images_list = images_list
        self.images_root = images_root
        self.masks_root = masks_root
        self.split = split
        self.training = training
        self.logger = logger
        self.edge = edge
        self.resize = resize
        self.num_classes = num_classes
        # Files
        self.id_to_trainid = domen1_id_to_trainid


    def id2trainId(self, label, reverse=False, ignore_label=255):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    def __getitem__(self, item):
        image_path = self.images_list[item]
        mask_path = os.path.join(self.masks_root, image_path[:-3]+'png')
        image_path = os.path.join(self.images_root, image_path)
        image = Image.open(image_path).convert("RGB")
        gt_image = Image.open(mask_path)
    
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

    def _train_sync_transform(self, img, mask):
        '''
        :param image:  PIL input image
        :param gt_image: PIL input gt_image
        :return:
        '''
        
        if True:
            # random mirror
            a = random.random()
            if a < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if mask:
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            crop_w, crop_h = self.resize
        
        if True:
            img = img.resize(self.resize, Image.BICUBIC)
            if mask:
                mask = mask.resize(self.resize, Image.NEAREST)

        if True:
            # gaussian blur as in PSP
            b = random.random()
            c = random.random()
            # print(a,b,c)
            if b < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=c))
        # final transform
        if mask:
            img = self._img_transform(img)
            mask, edge_mask = self._mask_transform(mask)
            
            return img, mask, edge_mask
        else:
            img = self._img_transform(img)
            return img

    
    def _val_sync_transform(self, img, mask):
        img = img.resize(self.resize, Image.BICUBIC)
        mask = mask.resize(self.resize, Image.NEAREST)
        # final transform
        img = self._img_transform(img)
        mask, edge_mask = self._mask_transform(mask)
        return img, mask, edge_mask    


    def _mask_transform(self, gt_image):
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        #target = np.expand_dims(target, axis=0)
        edge_mask = np.nan
        if self.edge:
            edge_mask = F.mask_to_binary_edge(
                target, radius=1, num_classes=self.num_classes)

        return target, edge_mask
    
    def _img_transform(self, image):
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)

        new_image = image

        return new_image
    

    def __len__(self):
        return len(self.images_list)
