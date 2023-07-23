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
from .mmseg.models.sam.image_encoder_anchor import ImageEncoderViT_anchor
from .mmseg.models.sam.image_encoder_original import ImageEncoderViT_original
from .roi_head_custom import RoIHeads_custom
from .transforms import ResizeLongestSide

logger = logging.getLogger(__name__)
from .sam import SAM, _iou_loss
from collections import OrderedDict

@register('sam_original')
class SAM_original(SAM):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__(inp_size=inp_size, encoder_mode=encoder_mode, loss=loss)
        self.model_name = 'sam_original'
        self.image_encoder=ImageEncoderViT_original(
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
        print("sam_original init")
