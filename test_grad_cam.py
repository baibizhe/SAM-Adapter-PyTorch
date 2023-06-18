import os

#@title CLIP-Loc settings
clip_version = "ViT-B/32" #@param ["RN50x16", "RN50x4", "RN50", "RN101", "ViT-B/32", "ViT-B/16", "hila"]
cam_version  = 'gScoreCAM' #@param ['GradCAM', 'ScoreCAM', 'GracCAM++', 'AblationCAM', 'XGradCAM', 'EigenCAM', 'EigengradCAM', 'LayerCAM', 'HilaCAM', 'GroupCAM', 'SSCAM1', 'SSCAM2', 'RawCAM', 'GradientCAM', 'gScoreCAM']
#@title ## Markdown

#@markdown Top-k channels used in gScoreCAM (default=100):
topk_channels = 216 #@param {type:"slider", min:1, max:3072, step:1}
cam_version  = cam_version.lower()
is_transformer = 'vit' in clip_version.lower()

from gScoreCAM.model_loader.clip_loader import load_clip
from gScoreCAM.tools.cam import CAMWrapper

clip_model, preprocess, target_layer, cam_trans, clip = load_clip(clip_version)
cam_wrapper = CAMWrapper(clip_model,
                         preprocess=preprocess,
                         target_layers=[target_layer],
                         tokenizer=clip.tokenize,
                         drop=True,
                         cam_version=cam_version,
                         topk=topk_channels,
                         channels=None,
                         is_transformer=is_transformer,
                         cam_trans=cam_trans)

def get_visualization(clip_model, cam_wrapper, img, prompt):
  # encode image and prompt
  raw_size    = img.size
  input_img   = preprocess(img).unsqueeze(0).cuda()
  text_token  = clip.tokenize(prompt).cuda()
  clip_logits = clip_model(input_img.cuda(), text_token.cuda())
  # get cam for prompt and overlay on input image
  cam = cam_wrapper.getCAM(input_img, text_token, raw_size, 0)

  float_img = img_as_float(img)
  if len(float_img.shape) == 2:
      float_img = color.gray2rgb(float_img)
  cam_img   = show_cam_on_image(float_img, cam, use_rgb=True)
  cam_img   = Image.fromarray(cam_img)
  cat_img = Image.new('RGB', (raw_size[0]*2, raw_size[1]))
  cat_img.paste(img, (0,0))
  cat_img.paste(cam_img, (raw_size[0],0))
  score = clip_logits[0].detach().item()
  return cat_img, score

#@title Play with CLIP
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import requests
import urllib.request
from PIL import Image
from skimage.util import img_as_float
from skimage import color
from gScoreCAM.pytorch_grad_cam.utils.image import show_cam_on_image
from IPython.display import display
import gc
import torch
import validators
gc.collect()
with torch.no_grad():
    torch.cuda.empty_cache()# get image
image = "/home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/images/cju0qx73cjw570799j4n5cjze.jpg"
img = Image.open(image)
# get prompt
prompt = "a t-shirt" #@param {type: "string"}

visualization, score = get_visualization(clip_model, cam_wrapper, img, prompt)

display(visualization)
print({'CLIP score':  score, 'Prompt': prompt})