
import functools
import random
import math
import warnings

from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

from datasets import register
import cv2
from math import pi
from torchvision.transforms import InterpolationMode
from torchvision.ops import masks_to_boxes

#
# import torch.nn.functional as F
# def to_mask(mask):
#     return transforms.ToTensor()(
#         transforms.Grayscale(num_output_channels=1)(
#             transforms.ToPILImage()(mask)))
#
#
# def resize_fn(img, size):
#     return transforms.ToTensor()(
#         transforms.Resize(size)(
#             transforms.ToPILImage()(img)))
#
#
# @register('val')
# class ValDataset(Dataset):
#     def __init__(self, dataset, inp_size=None, augment=False):
#         self.dataset = dataset
#         self.inp_size = inp_size
#         self.augment = augment
#         warnings.warn('this is val folder')
#
#         self.img_transform = transforms.Compose([
#                 transforms.Resize((inp_size, inp_size)),
#
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#         self.mask_transform = transforms.Compose([
#             transforms.Resize((inp_size, inp_size)),
#             transforms.ToTensor(),
#             ])
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         img, mask = self.dataset[idx]
#         return {
#             'inp': self.img_transform(img),
#             'gt': self.mask_transform(mask),
#         }
#
#
# @register('train')
# class TrainDataset(Dataset):
#     def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
#                  augment=False, gt_resize=None):
#         self.dataset = dataset
#         self.size_min = size_min
#         if size_max is None:
#             size_max = size_min
#         self.size_max = size_max
#         self.augment = augment
#         self.gt_resize = gt_resize
#
#         self.inp_size = inp_size
#         self.normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])])
#         self.img_transform = transforms.Compose([
#                 transforms.Resize((self.inp_size, self.inp_size)),
#                 transforms.RandomRotation(90),
#                 transforms.RandomChoice([
#                     transforms.RandomVerticalFlip(),
#                     transforms.RandomHorizontalFlip(),
#                 ]),
#                 transforms.ToTensor(),
#             ])
#         # self.inverse_transform = transforms.Compose([
#         #         transforms.Normalize(mean=[0., 0., 0.],
#         #                              std=[1/0.229, 1/0.224, 1/0.225]),
#         #         transforms.Normalize(mean=[-0.485, -0.456, -0.406],
#         #                              std=[1, 1, 1])
#         #     ])
#         self.mask_transform = transforms.Compose([
#                 transforms.Resize((self.inp_size, self.inp_size)),
#                 transforms.ToTensor(),
#             ])
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         img, mask = self.dataset[idx]
#
#         # random filp
#         if random.random() < 0.5:
#             img = img.transpose(Image.FLIP_LEFT_RIGHT)
#             mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
#
#
#         img = transforms.Resize((self.inp_size, self.inp_size))(img)
#         mask = transforms.Resize((self.inp_size, self.inp_size), interpolation=InterpolationMode.NEAREST)(mask)
#
#         return {
#             'inp': self.normalize(self.img_transform(img)),
#             'gt': self.img_transform(mask)
#         }

import functools
import random
import math
import warnings

from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

from datasets import register
import cv2
from math import pi
from torchvision.transforms import InterpolationMode

import torch.nn.functional as F
def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        warnings.warn('this is val folder')

        self.img_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        mask = transforms.Resize((self.inp_size, self.inp_size), interpolation=InterpolationMode.NEAREST)(mask)
        mask = np.array(mask)
        if mask.max()!=1:
            mask=np.where(mask==0,0,1)
        mask = torch.tensor(mask).float()
        if len(mask.shape) ==2:
            mask= mask.unsqueeze(0)
        img = self.img_transform(img)
        return {
            'inp': img,
            'gt': mask,
        }


@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, gt_resize=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize

        self.inp_size = inp_size
        self.normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        warnings.warn('using more augmentation')
        self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.GaussianBlur((75, 75), sigma=(0.001, 2.0)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
                ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])

        # self.mask_transform = transforms.Compose([
        #         transforms.ToTensor(),
        #     ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        # random filp
        if random.random() < 0.5:
            rotate_ang = np.random.randint(0,90)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            img = img.rotate(rotate_ang)
            mask = mask.rotate(rotate_ang)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        img = transforms.Resize((self.inp_size, self.inp_size))(img)
        mask = transforms.Resize((self.inp_size, self.inp_size), interpolation=InterpolationMode.NEAREST)(mask)
        img= np.array(img)
        mask = np.array(mask)
        if mask.max()!=1:
            mask=np.where(mask==0,0,1)
        img = self.img_transform(img)
        mask = torch.tensor(mask).float()
        if len(mask.shape) ==2:
            mask= mask.unsqueeze(0)

        # print(mask.min(),mask.max(),'line4')
        # img = np.array(img)
        # mask = np.array(mask)
        # print(mask.sum())
        #
        # if mask.sum()==0:
        #     roi_input =inp
        #     roi_mask = gt
        #     x1, y1, x2, y2 = gt.shape
        #     roi_size = gt.shape
        # else:
        #     x1, y1, x2, y2 = masks_to_boxes(torch.tensor(np.array(mask)).unsqueeze(0))[0]
        #     x1, y1, x2, y2 =x1.int().item(), y1.int().item(), x2.int().item(), y2.int().item()
        #     # print(x1, y1, x2, y2)
        #     # roi_input = inp[:,x1:x2,y1:y2]
        #     # roi_mask = gt[:,x1:x2,y1:y2]
        #     roi_input = inp[:,y1:y2,x1:x2]
        #     roi_mask = gt[:,y1:y2,x1:x2]
        #     roi_size = roi_mask.shape
        #     roi_input = transforms.Resize((self.inp_size, self.inp_size))(roi_input)
        #     roi_mask = transforms.Resize((self.inp_size, self.inp_size), interpolation=InterpolationMode.NEAREST)(roi_mask)
        # # import matplotlib.pyplot as plt
        # plt.imshow(inp.permute(1, 2, 0).numpy())
        # plt.show()
        # plt.imshow(gt.permute(1, 2, 0).numpy())
        # plt.show()
        # plt.imshow(self.dataset[idx][0])
        # plt.show()
        # plt.imshow(self.dataset[idx][1])
        # plt.show()
        # plt.imshow(roi_input.permute(1, 2, 0))
        # plt.show()
        # plt.imshow(roi_mask.permute(1, 2, 0))
        # plt.show()
        # print(img.mean(),img.max(),img.min())

        return {'inp':img,'gt':mask}

        # return {'inp':inp,'gt':gt,'roi_input':roi_input,'roi_mask':roi_mask,'roi_box':torch.tensor([x1, y1, x2, y2]),'roi_size':torch.tensor(roi_size)}