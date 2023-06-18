# define model
import glob

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from obj.eval_mask_rcnn_on_UDIATB.run import get_instance_segmentation_model
num_classes=2
model = get_instance_segmentation_model(num_classes, True, None)
model.to('cuda')
model.eval()
model.load_state_dict(torch.load('/home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/obj/output/e_100_obj.pth'))
model_total_params = sum(p.numel() for p in model.parameters())
model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))
all_imgs = glob.glob('/home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TestDataset/*/images/*')
for img_p in all_imgs:
    img = cv2.imread(img_p)
    img=cv2.resize(img,(1024,1024))
    img = to_tensor(img).cuda()
    result = model([img])[0]
    print(img.min(),img.mean())
    print(result['boxes'].shape)
