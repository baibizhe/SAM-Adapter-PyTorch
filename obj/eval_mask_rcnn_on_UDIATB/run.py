import os
import time
import numpy as np
import random
import torch
import argparse
import torchvision
from torchvision.models.detection import MaskRCNN,FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import yaml
import models

from obj.eval_mask_rcnn_on_UDIATB import utils

from obj.eval_mask_rcnn_on_UDIATB.datasets import USSegDataset
from obj.eval_mask_rcnn_on_UDIATB.backbone_utils import resnet18_fpn_backbone, resnet50_fpn_backbone, VIT_fpn_backbone
import obj.eval_mask_rcnn_on_UDIATB.utils
import obj.eval_mask_rcnn_on_UDIATB.transforms as T
from obj.eval_mask_rcnn_on_UDIATB.engine import train_one_epoch, evaluate




def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_instance_segmentation_model(num_classes, pretrained, state_dict):

    model= fasterrcnn_resnet50_fpn(pretrained=True)

    return model


def get_customized_model(num_classes, pretrained=False, state_dict=None):
    '''
    get a mask rcnn model
    parameters:
        num_classes: instance class, including background
        pretrained: whether to load ImageNet pretrained parameters
        state_dict: self/semi-supervised pretrained parameters path
    '''
    # backbone = resnet50_fpn_backbone(pretrained, state_dict)
    # backbone = resnet18_fpn_backbone(pretrained, state_dict)

    with open('/home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-anchor-b-kvasir-seg.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    sam = models.make(config['model']).cuda()
    # for name, para in sam.named_parameters():
    #     para.requires_grad_(False)
    for name, para in sam.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
    backbone =sam.image_encoder
    backbone.out_channels = 256

    # backbone =VIT_fpn_backbone(backbone,None)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0,1,2,3],
                                                    output_size=7,
                                                    sampling_ratio=2)
    model = FasterRCNN(backbone,min_size=1024,max_size=1024,
                       num_classes=num_classes,
                       box_roi_pool=roi_pooler
                       # rpn_anchor_generator=AnchorGenerator(sizes=((128, 256, 512),))
                       # box_roi_pool=None,
                       )
    # print(model)
    print('created model')
    return model



class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)
 
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc
 
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / np.maximum(self.confusionMatrix.sum(axis=1),1)
        return classAcc
 
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc
 
    # MIoU
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.maximum(np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix), 1)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU
 
    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask].astype(int) + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix
 
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
 
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

################################## define data augmentation ####################################
 
def get_transform(train):
    # Only flip the image 
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # transforms.append(T.Normalize())
    # transforms.append(T.Resize())

    return T.Compose(transforms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear evaluation')
    parser.add_argument('-m', '--model', default='ResNet')
    parser.add_argument('-d', '--depth', type=int, default=18)
    # parser.add_argument('-dd', '--dataset_dir', default='/home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split4951/fold1/P51', help='path of data')
    # parser.add_argument('-td', '--test_dataset_dir', default='/home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split4951/fold1/P49', help='path of data')
    # parser.add_argument('-dd', '--dataset_dir', default='/home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split4060/fold1/P40', help='path of data')
    # parser.add_argument('-td', '--test_dataset_dir', default='/home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split4060/fold1/P40', help='path of data')
    parser.add_argument('-dd', '--dataset_dir', default='/home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold1/P20', help='path of data')
    parser.add_argument('-td', '--test_dataset_dir', default='/home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold1/P80', help='path of data')
    # parser.add_argument('-dd', '--dataset_dir', default='/home/ubuntu/works/code/working_proj/segment-anything/data/endovis_instrument_data/endovis19/traning/split2080/fold1/P20', help='path of data')
    # parser.add_argument('-td', '--test_dataset_dir', default='/home/ubuntu/works/code/working_proj/segment-anything/data/endovis_instrument_data/endovis19/test/Stage_3', help='path of data')

    parser.add_argument('-p', '--path', default='/home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/obj/eval_mask_rcnn_on_UDIATB/model_ckpt.zip', help='path of ckpt')
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-e', '--epoch', type=int, default=100)
    args = parser.parse_args()

    ###################################### dataset #########################################
    dataset_dir = args.dataset_dir
    print('data dir is:', dataset_dir)
    dataset = USSegDataset(dataset_dir, get_transform(train=True))
    dataset_test = USSegDataset(args.test_dataset_dir, get_transform(train=False),test=True)

    # split the dataset in train and test set
    set_seed(args.seed)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    ###################################### fine-tune the model #########################################
    state_dicts = [args.path]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
     
    # the dataset has two classes only - background and lesion
    num_classes = 2

    # get the model using the helper function
    for state_dict in state_dicts:

        print('\n\n'+'='*20, state_dict, '='*20+'\n\n')
        if state_dict != '':
            pretrained = True
        else:
            pretrained = False

        # define model
        model = get_customized_model(num_classes, True, None)
        model.to(device)

        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for name, para in model.named_parameters():
            if para.requires_grad ==   True:
                print(name)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, # 一般模型0.005学习率对于batchsize=1太大，容易loss=nan，但是学习率小了又会性能下降明显
                                    momentum=0.9, weight_decay=0.0005)
         
        # the learning rate scheduler decreases the learning rate by 10x every 10 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=10,
                                                       gamma=0.1)
         
        # training
        num_epochs = args.epoch
        start = time.time()
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            print('\nTrain epoch {}'.format(epoch))
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
         
            # update the learning rate
            lr_scheduler.step()
         
            # evaluate on the test dataset
            print('\nEvaluate epoch {}'.format(epoch))
            evaluate(model, data_loader_test, device=device,epoch=epoch)
            torch.save(model.state_dict(),os.path.join('/home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/obj/output',f'e_{epoch}_obj.pth'))
            # break
        print('Training finish, the time consumption is {}s'.format(round(time.time()-start)))

    # put the model in evaluation mode
    # model.eval()
    # metric = SegmentationMetric(2) # classes = bg, lesion
    # with torch.no_grad():
    #     for i, data in enumerate(data_loader_test):
    #         images, targets = data
    #         images = list(image.to(device) for image in images)
    #         #print(targets)
    #         targets = [np.array(t['masks']) for t in targets]
    #         targets = np.array(targets)
    #         targets = targets[0][0] > 0.5 # True/False
    #
    #         target_pred = model(images)  # (1,N,1,H,W), N is the num of detected tumors
    #         target_pred = np.array([np.array(t['masks'].cpu()) for t in target_pred])
    #         if target_pred.size == 0:  # no lesion detected, happens for tiny tumors
    #             print('No tumor detected!')
    #             target_pred = np.zeros_like(targets)
    #         else:
    #             target_pred = target_pred[0][0][0]
    #         try:
    #             target_pred = target_pred > 0.5 # True/False
    #             metric.addBatch(target_pred.astype(int), targets.astype(int))
    #         except:
    #             print(target_pred)
    #             print("skip a null prediction")
    #         # mask_pred = Image.fromarray(target_pred)
    #         # mask_pred.save("mask_pred{}.jpeg".format(i))
    #         # mask_true = Image.fromarray(targets)
    #         # mask_true.save("mask_true{}.jpeg".format(i))
    #
    # TP = metric.confusionMatrix[1][1]
    # FP = metric.confusionMatrix[0][1]
    # FN = metric.confusionMatrix[1][0]
    # TN = metric.confusionMatrix[0][0]
    # print(metric.confusionMatrix)
    # print('Dice: {}'.format(round(2*TP / (FP+2*TP+FN), 4)))
    # print('PPV: {}'.format(round(TP / (FP+TP), 4)))
    # print('Sensitivity: {}'.format(round(TP / (FN+TP), 4)))
    # mask_pred = Image.fromarray(target_pred)
    # mask_pred.save("mask_pred.jpeg")
    # mask_true = Image.fromarray(targets)
    # mask_true.save("mask_true.jpeg")
    
    
    
    
    
    
    
    
    
    
    
