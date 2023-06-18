import cv2
from scipy import ndimage

# import open_clip
# import torch
# import os
# import torch
# import unittest
# import numpy as np
# from PIL import Image as PilImage
# from omnixai.data.text import Text
# from omnixai.data.image import Image
# from omnixai.data.multi_inputs import MultiInputs
# from omnixai.preprocessing.image import Resize
# from omnixai.explainers.vision_language.specific.gradcam import GradCAM
#
# model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
# tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
# x= torch.ones(size=(1,3,224,224))
# # output = model(x,tokenizer('poly'))
# # output2 = model(x,tokenizer('a image of poly'))
# # output3 = model(x,tokenizer('an image of poly'))
#
# image = Resize(size=480).transform(
#     Image(PilImage.open("/home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/images/ckcu9ij2e00063b5yrrbb3f2o.jpg").convert("RGB")))
# text = Text("an image of poly")
# inputs = MultiInputs(image=image, text=text)
# image_processor = model.visual
# text_processor = model.text
#
# def preprocess(x: MultiInputs):
#     images = torch.stack([image_processor(z.to_pil()) for z in x.image])
#     texts = [text_processor(z) for z in x.text.values]
#     return images, texts
# explainer = GradCAM(
#     model=model,
#     # target_layer=model.text_encoder.base_model.base_model.encoder.layer[6].
#     #     crossattention.self.attention_probs_layer,
#     target_layer=model.text.transformer.encoder.layer[-1].attention.output.LayerNorm,
#     preprocess_function=preprocess,
#     tokenizer=tokenizer,
#     loss_function=lambda outputs: outputs[:, 1].sum()
# )
# explanations = explainer.explain(inputs)
# explanations.ipython_plot()
import torch
import matplotlib.pyplot as plt
def fft( x, rate):
    # the smaller rate, the smoother; the larger rate, the darker
    # rate = 4, 8, 16, 32
    mask = torch.zeros(x.shape).to(x.device)
    w, h = x.shape[-2:]
    line = int((w * h * rate) ** .5 // 2)
    mask[:, w // 2 - line:w // 2 + line, h // 2 - line:h // 2 + line] = 1

    fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
    # mask[fft.float() > self.freq_nums] = 1
    # high pass: 1-mask, low pass: mask
    fft = fft * (1 - mask)
    # fft = fft * mask
    fr = fft.real
    fi = fft.imag

    fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
    inv = torch.fft.ifft2(fft_hires, norm="forward").real

    inv = torch.abs(inv)

    return inv

img_path ='/home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/images/ckcud94xp000h3b5yael7l26v.jpg'
# mask_path ='/home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/data/Kvasir-SEG/masks/cju88v2f9oi8w0871hx9auh01.jpg'
mask_path ='/home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/labels/ckcud94xp000h3b5yael7l26v.png'

image =cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = torch.tensor(image)
imagefft = fft(image,0.25)
plt.figure(figsize=(30,30))
plt.subplot(3,3,1)
plt.axis('off')
plt.imshow(image)
plt.subplot(3,3,2)
plt.axis('off')
plt.imshow(imagefft[:,:,0])
plt.subplot(3,3,3)
plt.axis('off')
plt.imshow(imagefft[:,:,1])
plt.subplot(3,3,4)
plt.imshow(imagefft[:,:,2])
plt.axis('off')
plt.subplot(3,3,5)
plt.imshow(cv2.imread(mask_path))
plt.subplot(3,3,6)
plt.imshow(imagefft[:])
plt.subplot(3,3,7)
sobel_result = ndimage.sobel(image)
plt.imshow(sobel_result[:,:,0])
plt.subplot(3,3,8)
plt.imshow(sobel_result[:,:,1])
plt.subplot(3,3,9)
plt.imshow(sobel_result[:,:,2])
plt.axis('off')
plt.tight_layout()
plt.savefig('visualize_of_fft_kvsair_instrument.png')
plt.show()

