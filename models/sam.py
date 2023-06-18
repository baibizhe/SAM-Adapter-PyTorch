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

from models import register
from obj.eval_mask_rcnn_on_UDIATB.run import get_instance_segmentation_model
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer
from .mmseg.models.sam.image_encoder_adapter import ImageEncoderViT_adapter
from .transforms import ResizeLongestSide

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple
from torchvision.ops import masks_to_boxes
import cv2

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss




def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


def inverse_normalize(tensor: torch.Tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> torch.Tensor:
    mean = torch.tensor(mean).view(3, 1, 1).cuda()
    std = torch.tensor(std).view(3, 1, 1).cuda()
    return tensor * std + mean




@register('sam')
class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.image_encoder = ImageEncoderViT(
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
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.boxes = nn.Parameter(torch.tensor([[0.,0.,1024.,1024.]]))
        self.obj_transform = get_transform(True)
        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, self.prompt_embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)

        # self.obj_model = get_instance_segmentation_model(2,True, None)
        # self.obj_model.to('cuda')
        # self.obj_model.load_state_dict(torch.load('/home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/obj/output/e_183_obj.pth'))
        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()
        print('prompt_embed_dim] // 2',encoder_mode['prompt_embed_dim'] // 2)
        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])
        self.dice_focal_loss = monai.losses.DiceLoss(to_onehot_y=True)
        self.transform = ResizeLongestSide(inp_size)
    def set_input(self, input, gt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)
    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, (1024,1024))
        # print(corner_embedding.shape)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding


    def forward(self):
        bs = 1
        # print(self.boxes)
        # Embed prompts
        # _,H,W = self.gt_mask[0].shape
        #
        # x0,y0,x1,y1 =masks_to_boxes(self.gt_mask[0])[0].cpu().numpy().astype('int')
        # x0 = max(0, x0 - np.random.randint(0, 100))
        # y0 = min(W, y0 + np.random.randint(0, 100))
        # x1 = max(0, x1 - np.random.randint(0, 100))
        # y1 = min(H, y1 + np.random.randint(0, 100))
        # boxes = torch.tensor([[x0,y0,x1,y1]])
        # print(boxes.shape,215)

        # with torch.no_grad():
            # self.obj_model.eval()
            # img_inverse_n = inverse_normalize(self.input[0])
            # result = self.obj_model([img_inverse_n])[0]
            # if len(result['boxes'])==0:
            #     boxes_highest= torch.tensor([[0,0,1024,1024]]).cuda()
            # else:
            #     boxes_highest = result['boxes'][result['scores'].argmax().item()]

            # print(221,boxes_highest)
            # x0, y0, x1, y1 = boxes_highest
            # x0, y0, x1, y1 =int(x0),int(y0),int(x1),int(y1)
            # boxes= boxes_highest.unsqueeze(0)
            # print(boxes.shape)

        # draw_img = self.gt_mask.detach()
        # draw_img = draw_img.clone().cpu().numpy()[0].reshape(1024,1024,1)
        # draw_img = numpy.repeat(draw_img,3,2)
        # cv2.rectangle(draw_img, (x0,y0), (x1,y1), color=(255, 255, 0), thickness=2)
        # plt.imshow(draw_img)
        # plt.show()
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        box_embeddings = self._embed_boxes(torch.tensor(self.boxes,device=self.device))
        sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(self.input)

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

    def infer(self, input):
        bs = 1
        # Embed prompts
        sparse_embeddings = self.get_sparse_emb(bs)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        self.features = self.image_encoder(input)

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
        return masks

    def get_sparse_emb(self, bs):
        # x0, y0, x1, y1 = masks_to_boxes(gt_mask[0])[0].cpu().numpy().astype('int')
        # _,H,W = self.gt_mask[0].shape
        # x0 = max(0, x0 - np.random.randint(0, 100))
        # y0 = min(W, y0 + np.random.randint(0, 100))
        # x1 = max(0, x1 - np.random.randint(0, 100))
        # y1 = min(H, y1 + np.random.randint(0, 100))
        # draw_img = gt_mask.detach()
        # draw_img = draw_img.clone().cpu().numpy()[0].reshape(1024,1024,1)
        # draw_img = numpy.repeat(draw_img,3,2)
        # cv2.rectangle(draw_img, (x0,y0), (x1,y1), color=(255, 255, 0), thickness=2)
        # plt.imshow(draw_img)
        # plt.show()
        # with torch.no_grad():
        #     self.obj_model.eval()
        #
        #     img_inverse_n = inverse_normalize(self.input[0])
        #     # print(img_inverse_n.shape)
        #     result = self.obj_model([img_inverse_n])[0]
        #     if len(result['boxes'])==0:
        #         boxes_highest= torch.tensor([[0,0,1024,1024]]).cuda()
        #     else:
        #         boxes_highest = result['boxes'][result['scores'].argmax().item()]
        #     boxes= boxes_highest.unsqueeze(0)
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.device)
        box_embeddings = self._embed_boxes(torch.tensor(self.boxes, device=self.device))
        sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        return sparse_embeddings

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def backward_G(self):
        self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_G += _iou_loss(self.pred_mask, self.gt_mask)
            # self.loss_G += self.dice_focal_loss(self.pred_mask, self.gt_mask)*0.5

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
def get_transform(train):
    # Only flip the image
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())

    return Compose(transforms)
@register('sam_adapter')
class SAM_adapter(SAM):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__(inp_size=inp_size, encoder_mode=encoder_mode, loss=loss)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.image_encoder = ImageEncoderViT_adapter(
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
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "mask_decoder" not in k and "prompt_encoder" not in k:
                    p.requires_grad = False

        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()

        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])