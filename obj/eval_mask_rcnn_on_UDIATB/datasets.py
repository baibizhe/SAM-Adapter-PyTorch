import glob
import os
import torch
import numpy as np
import torch.utils.data
import torchvision
from PIL import Image
from PIL.Image import Resampling


class USSegDataset(torch.utils.data.Dataset):
    '''
    Dataset for Ultrasound segmentation. File structure：
    UDIAT_Dataset_B:
        GT: masks for each image，png file，bg: 0, lesion: 255
        original: data images，png
    '''
    def __init__(self, root, transforms=None,test=False,num_of_data=200):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "imgs"))))
        self.imgs = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), self.imgs))
        if test:
            self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))
            self.masks = list(map(lambda x:os.path.join(root,'masks',x),self.masks ))
            # self.masks = list(sorted(os.listdir(os.path.join(root, "labels"))))
            # self.masks = list(map(lambda x:os.path.join(root,'labels',x),self.masks ))

        else:
             self.masks = list(sorted(os.listdir(os.path.join(root, "labels"))))
             self.masks = list(map(lambda x:os.path.join(root,'labels',x),self.masks ))

        self.masks = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), self.masks))
        if num_of_data:
            self.imgs = self.imgs[0:num_of_data]
            self.masks = self.masks[0:num_of_data]
        self.imgs = list(map(lambda x:os.path.join(root,'imgs',x),self.imgs ))

        # if test:
        #     self.imgs = glob.glob('/home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TestDataset/*/images/*')
        #     self.masks = glob.glob('/home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TestDataset/*/masks/*')
        #     self.imgs = sorted(self.imgs)
        #     self.masks = sorted(self.masks)

    def __getitem__(self, idx):
        # load images and masks
        # img_path = os.path.join(self.root, "imgs", self.imgs[idx])
        # mask_path = os.path.join(self.root, "labels", self.masks[idx])
        img_path =  self.imgs[idx]
        mask_path =  self.masks[idx]
        img = Image.open(img_path).convert("RGB").resize((1024,1024))
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance with 0 being background
        mask = Image.open(mask_path).resize((1024,1024),Resampling.NEAREST)
 
        mask = np.array(mask)
        if mask.max()!=1:
            mask=np.where(mask==0,0,1)
        mask=(mask>0)*255 # ensure 0,255
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
 
        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
 
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        if len(mask.shape)==3:
            mask = mask[:,:,0]
        # print(mask.shape)

        boxes = torchvision.ops.masks_to_boxes(torch.tensor(mask).unsqueeze(0))[0]
        # print(boxes)
        boxes=boxes.unsqueeze(0)
        # for i in range(num_objs):
            # boundary of masks as bbox
            # pos = np.where(masks[i])
            # xmin = np.min(pos[1])
            # xmax = np.max(pos[1])
            # ymin = np.min(pos[0])
            # ymax = np.max(pos[0])
            # # boxes.append([xmin, ymin, xmax, ymax])
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)















