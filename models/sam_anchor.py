import logging
import warnings
from functools import partial

import matplotlib.pyplot as plt
import monai.losses
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmcv.cnn import ConvModule
import einops
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from models import register
from obj.eval_mask_rcnn_on_UDIATB.run import get_instance_segmentation_model
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer
from .mmseg.models.sam.image_encoder_adapter import ImageEncoderViT_adapter
from .mmseg.models.sam.image_encoder_anchor import ImageEncoderViT_anchor
from .roi_head_custom import RoIHeads_custom
from .transforms import ResizeLongestSide

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple
from torchvision.ops import masks_to_boxes
import cv2
from .sam import SAM, _iou_loss
from collections import OrderedDict
from torch.cuda.amp import autocast as autocast

class SAMAggregatorNeck(nn.Module):
    def __init__(
            self,
            in_channels=[1280]*16,
            inner_channels=128,
            selected_channels: list=None,
            out_channels=256,
            kernel_size=3,
            stride=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
            up_sample_scale=4,
            init_cfg=None,
            **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.out_channels = out_channels
        self.stride = stride
        self.selected_channels = selected_channels
        self.up_sample_scale = up_sample_scale

        self.down_sample_layers = nn.ModuleList()
        for idx in self.selected_channels:
            self.down_sample_layers.append(
                nn.Sequential(
                    ConvModule(
                        in_channels[idx],
                        inner_channels,
                        kernel_size=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                    ConvModule(
                        inner_channels,
                        inner_channels,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                )
            )
        self.fusion_layers = nn.ModuleList()
        for idx in self.selected_channels:
            self.fusion_layers.append(
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        self.up_layers = nn.ModuleList()
        self.up_layers.append(
            nn.Sequential(
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ),
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        )
        self.up_layers.append(
            ConvModule(
                inner_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None
            )
        )

        self.up_sample_layers = nn.ModuleList()
        assert up_sample_scale == 4
        self.up_sample_layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        )

        self.up_sample_layers.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.up_sample_layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        )

        self.up_sample_layers.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, inputs):
        inner_states = inputs
        inner_states = [einops.rearrange(inner_states[idx], 'b h w c -> b c h w') for idx in self.selected_channels]
        inner_states = [layer(x) for layer, x in zip(self.down_sample_layers, inner_states)]

        x = None
        for inner_state, layer in zip(inner_states, self.fusion_layers):
            if x is not None:
                inner_state = x + inner_state
            x = inner_state + layer(inner_state)
        x = self.up_layers[0](x) + x
        img_feats_0 = self.up_layers[1](x)

        img_feats_1 = self.up_sample_layers[0](img_feats_0) + self.up_sample_layers[1](img_feats_0)

        img_feats_2 = self.up_sample_layers[2](img_feats_1) + self.up_sample_layers[3](img_feats_1)

        return img_feats_2, img_feats_1, img_feats_0
class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Args:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> 'ImageList':
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)

@register('sam_anchor')
class SAM_anchor(SAM):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__(inp_size=inp_size, encoder_mode=encoder_mode, loss=loss)
        self.model_name = 'sam_anchor'
        self.image_encoder=ImageEncoderViT_anchor(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )
        print("sam_anchor init")
        self.obj_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0,1,2,3],
                                                    output_size=7,
                                                    sampling_ratio=2)
        anchor_sizes = ((32,), (64,), (128,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.obj_rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )
        self.scaler = torch.cuda.amp.GradScaler()

        out_channels=256
        rpn_pre_nms_top_n_train = 2000
        rpn_pre_nms_top_n_test = 1000
        rpn_post_nms_top_n_train = 2000
        rpn_post_nms_top_n_test = 1000
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.obj_rpn_head = RPNHead(
            out_channels, self.obj_rpn_anchor_generator.num_anchors_per_location()[0]
        )
        self.obj_rpn = RegionProposalNetwork(
            anchor_generator=self.obj_rpn_anchor_generator,
            head=self.obj_rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=rpn_pre_nms_top_n,
            post_nms_top_n=rpn_post_nms_top_n,
            nms_thresh=0.7,
            score_thresh=0.0)
        in_channels, selected_channels = self.get_embed_dim_selected_channels(encoder_mode)
        self.obj_neck = SAMAggregatorNeck(
            in_channels=in_channels,
            inner_channels=32,
            selected_channels=selected_channels,
            out_channels=256,
            up_sample_scale=4,)

        resolution = self.obj_roi_pooler.output_size[0]
        representation_size = 1024
        self.obj_box_head = TwoMLPHead(out_channels * resolution ** 2,representation_size)
        self.obj_box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes=2)
        self.obj_roi_heads = RoIHeads_custom(
            box_roi_pool=self.obj_roi_pooler, box_head=self.obj_box_head, box_predictor=self.obj_box_predictor,
            fg_iou_thresh=0.5, bg_iou_thresh=0.5,
            batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100)
        # self.obj_roi_heads = RoIHeads(
        #     box_roi_pool=self.obj_roi_pooler, box_head=self.obj_box_head, box_predictor=self.obj_box_predictor,
        #     fg_iou_thresh=0.5, bg_iou_thresh=0.5,
        #     batch_size_per_image=512, positive_fraction=0.25,
        #     bbox_reg_weights=None,
        #     score_thresh=0.05,
        #     nms_thresh=0.5,
        #     detections_per_img=100)
        # self.rcnn_transform = GeneralizedRCNNTransform(1024, 1024,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  # do not normalize here because we already normalize in dataset
        self.loss_info_dict ={}                                                           # return (image - mean[:, None, None]) / std[:, None, None]


        self.rcnn_transform = GeneralizedRCNNTransform(1024, 1024,[0,0,0],[1,1,1])  # do not normalize here because we already normalize in dataset


    def get_embed_dim_selected_channels(self, encoder_mode):
        if encoder_mode['embed_dim'] == 768:
            print('sam base init with embed_dim  768')
            in_channels = [768] * 12
            selected_channels = list(range(4, 12, 2))
        elif encoder_mode['embed_dim'] == 1024:
            print('sam l init with embed_dim  1024')
            in_channels = [1024] * 32
            selected_channels = list(range(8, 32, 2))
        elif encoder_mode['embed_dim'] == 1280:
            print('sam huge init with embed_dim  1280')
            in_channels = 1024
            selected_channels = list(range(8, 32, 2))
        else:
            raise Exception('others does not support yet')
        return in_channels, selected_channels

    def forward(self,temperature=1):
        bs = 1
        _, H, W = self.gt_mask[0].shape

        self.features, features_multilayers = self.image_encoder(self.input)
        fused_features_multilayers = self.obj_neck(features_multilayers)
        fused_features_multilayers = OrderedDict([(0, fused_features_multilayers[0]),
                                (1, fused_features_multilayers[1]),
                                (2, fused_features_multilayers[2]),
                                ])
        # for key, value in fused_features_multilayers.items():
        #     warnings.warn(str(key)+ str(value.shape))
        images_list = []
        for i in range(self.input.shape[0]):
            images_list.append(self.input[i])
        # for i in images_list:
        #     print(i.shape)
        # print(self.rcnn_targets,329)
        images, targets = self.rcnn_transform(images_list, self.rcnn_targets)


        proposals, proposal_losses = self.obj_rpn(images, fused_features_multilayers, targets)
        detections, detector_losses = self.obj_roi_heads(fused_features_multilayers, proposals, images.image_sizes, targets)
        detections = self.rcnn_transform.postprocess(detections, images.image_sizes, images.image_sizes)
        # print(detections)

        self.obj_losses = {}
        self.obj_losses.update(detector_losses)
        self.obj_losses.update(proposal_losses)

        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        # box_embeddings = self._embed_boxes(torch.tensor(boxes_highest, device=self.device))
        # sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        self.pred_mask = masks
    def set_input_anchor(self, input, gt_mask,rcnn_targets):
        rcnn_targets = self.dict_to_list_of_dicts(rcnn_targets)
        self.input = input
        self.gt_mask = gt_mask
        self.rcnn_targets = rcnn_targets
    def backward_G(self):
        self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_G += _iou_loss(self.pred_mask, self.gt_mask)
        self.loss_info_dict.update({"sam loss":self.loss_G.item() })

        obj_losses_reduced = sum(loss for loss in self.obj_losses.values())
        self.loss_G+=obj_losses_reduced
        self.loss_G.backward()

        #
        self.loss_info_dict.update({"det loss":obj_losses_reduced.item() })
        for key, value in self.obj_losses.items():
            # Check if the value is a tensor
            if torch.is_tensor(value):
                # Extract the scalar value
                scalar_value = value.item()
                # Create a new key with the original key name
                new_key = key
                self.loss_info_dict[new_key] = scalar_value

        self.loss_info_dict.update(self.obj_losses)
        self.loss_info_dict['total loss'] = self.loss_G
        # print(self.loss_info_dict)
    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()

        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def infer(self, input,tempature=1,gt_original_cpu=None):
        assert input.shape[0] ==1,'current only support batch =1'
        bs = 1
        # Embed prompts
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        features, features_multilayers = self.image_encoder(input)
        fused_features_multilayers = self.obj_neck(features_multilayers)
        fused_features_multilayers = OrderedDict([(0, fused_features_multilayers[0]),
                                                  (1, fused_features_multilayers[1]),
                                                  (2, fused_features_multilayers[2]),
                                                  ])
        for key, value in fused_features_multilayers.items():
            warnings.warn(str(key) + str(value.shape))
        images_list = []
        for i in range(input.shape[0]):
            images_list.append(input[i])

        images, targets = self.rcnn_transform(images_list, self.rcnn_targets)

        proposals, proposal_losses = self.obj_rpn(images, fused_features_multilayers, targets)
        detections, detector_losses = self.obj_roi_heads(fused_features_multilayers, proposals, images.image_sizes,
                                                         targets)
        detections = self.rcnn_transform.postprocess(detections, images.image_sizes, images.image_sizes)
        boxes_highest =[]
        # print(detections)
        for i in range(self.input.shape[0]):
            if len(detections[i]['boxes']) ==0:
                sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
            elif detections[i]['scores'].max() < 0.6:
                sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
            else:
                boxes_highest.append(detections[i]['boxes'][detections[i]['scores'].argmax().item()].unsqueeze(0))
                boxes_highest = torch.cat(boxes_highest,0).cuda()
                sparse_embeddings = self.get_sparse_emb(bs,boxes_highest,gt_original_cpu)

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks

    def get_sparse_emb(self, bs, boxes_highest,gt_original_cpu=None):
        # x0, y0, x1, y1 = masks_to_boxes(gt_mask[0])[0].cpu().numpy().astype('int')
        # _,H,W = self.gt_mask[0].shape
        # x0 = max(0, x0 - np.random.randint(0, 100))
        # y0 = min(W, y0 + np.random.randint(0, 100))
        # x1 = max(0, x1 - np.random.randint(0, 100))
        # y1 = min(H, y1 + np.random.randint(0, 100))

        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.device)
        box_embeddings = self._embed_boxes(torch.tensor(boxes_highest, device=self.device))
        # print(box_embeddings)
        # print(box_embeddings.shape)

        sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        # if gt_original_cpu != None:
        #     draw_img = gt_original_cpu.detach()
        #     draw_img = draw_img.clone().cpu().numpy()[0].reshape(1024,1024,1)
        #     if draw_img.shape[2] ==1:
        #         draw_img = numpy.repeat(draw_img,3,2)
        #     x0, y0, x1, y1 = boxes_highest[0]
        #     x0, y0, x1, y1 =int(x0),int(y0),int(x1),int(y1)
        #     cv2.rectangle(draw_img, (x0,y0), (x1,y1), color=(255, 255, 0), thickness=2)
        #     plt.imshow(draw_img)
        #     plt.show()
        return sparse_embeddings
    def dict_to_list_of_dicts(self,input_dict):
        # Find the length of the lists in the input dictionary
        list_length = len(next(iter(input_dict.values())))

        # Create an empty list to store the dictionaries
        list_of_dicts = []

        # Iterate through the lists and create a new dictionary for each combination
        for i in range(list_length):
            new_dict = {}
            for key, value_list in input_dict.items():
                new_dict[key] = value_list[i]
                if isinstance(new_dict[key], torch.Tensor):
                    new_dict[key] = new_dict[key].cuda()
            list_of_dicts.append(new_dict)

        return list_of_dicts