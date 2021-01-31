# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        # the path to kitti raw(containing calib files)
        self.data_path = data_path
        # ALL the text lines content in split train or val .txt files
        self.filenames = filenames
        # the height and width set to adjust to the input of network
        self.height = height
        self.width = width
        # the scales of resized image,different input will generate different disparity map which would resized to 192*640 size
        self.num_scales = num_scales
        # the filter to process the image during resizing
        self.interp = Image.ANTIALIAS
        # [0,-1,1]
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        # define the type of input image,png or jpeg
        self.img_ext = img_ext
        # PIL.Image.open and convert to rgb
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            # why not in transforms.Compose????????
            # transforms.ColorJitter.get_params(
            #     self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)


        # check the existence of depth data
        self.load_depth = self.check_depth()


        # DO NOT KNOW WHY THE VALUE OF K COULD BE SET AS BELOW?????????????????????????????????????????
        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)




    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """

        # resize the image size to (640,192),(320,96),(160,48),(80,24)
        for k in list(inputs):
            # print(k)
            # frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    # here the lower resolution image is resized by previous image
                    # would it be better to resize the image all from original one???????????????????????
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        # until this step,the content of input is the image of 5 size and corresponding K and inv_K(without K of 1242*375 size)

        # color aug and transform image to tensor
        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and      ?????????????????????????
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        # ['2011_09_26\\2011_09_26_drive_0048_sync', '5', 'l']
        line = self.filenames[index].split()
        # 2011_09_26\\2011_09_26_drive_0048_sync
        folder = line[0]

        if len(line) == 3:
            # the image index in kitti raw dataset,such as (000000000+frame_index).png
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            # the left image or right image
            side = line[2]
        else:
            side = None


        # get the color image of original size(375*1242) and finish flip process,totally 3 data got:
        # inputs[("color", 0, -1)]  0 response to the middle frame
        # inputs[("color", -1, -1)] -1 response to previous frame
        # inputs[("color", 1, -1)]  1 response to next frame
        for i in self.frame_idxs:
            if i == "s":
                # if the input image is right side(r),its opposite side is left
                other_side = {"r": "l", "l": "r"}[side]
                # load corresponding rgb image and finish flip
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        # resize and color augmentation
        self.preprocess(inputs, color_aug)


        # delete the original image,which would not be used in later training
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]


        if self.load_depth:
            # get the depth map of original shape and finish flip
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)   #375*1242
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)    #1*375*1242
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        # DO NOT KNOW THE USE OF inputs["stereo_T"]?????????????????????????????????????????????
        # print(inputs["stereo_T"])

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
