import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
import numpy as np
import cv2
import os
import random

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        try:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            try:
                loss_dict = model(images, targets)
            except :
                print('loss exception')
                continue
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            # break
        except Exception as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def combine_images(images, cols, rows):
    # Calculate the dimensions of the combined image
    max_width = max(image.shape[1] for image in images)
    max_height = max(image.shape[0] for image in images)
    combined_width = max_width * cols
    combined_height = max_height * rows

    # Create an empty image with the combined dimensions
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Copy each processed image into the combined image
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        y_offset = row * max_height
        x_offset = col * max_width
        if i ==9:
            break
        combined_image[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image

    return combined_image
def draw_boxes(image, detection_result, scores):
    # Convert the tensor to a NumPy array
    boxes = detection_result.numpy()
    scores = scores.numpy()

    # Iterate through the boxes and draw them on the image
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        # Generate a random color for each box
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Draw the bounding box on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Display the score on the image
        text = f"{scores[i]:.2f}"
        cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image
@torch.no_grad()
def evaluate(model, data_loader, device,epoch):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    processed_images = []
    plot_flat = False
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        if len(processed_images) <= 9:
            detection_results = res[list(res.keys())[0]]['boxes']
            scores_list= res[list(res.keys())[0]]['scores']
            image_with_boxes = draw_boxes(targets[0]['masks'].cpu().numpy().copy()*255 , detection_results,
                                          scores_list)
            if image_with_boxes.shape[0]==1:
                image_with_boxes = np.repeat(np.expand_dims(image_with_boxes[0],2),3,2)
            # print(image_with_boxes.shape)
            processed_images.append(image_with_boxes)
        else:
            if not plot_flat:
                cols = 3  # The number of columns for the combined image
                rows = 3  # The number of rows for the combined image
                combined_image = combine_images(processed_images, cols, rows)
                plot_flat=True
                # Save the combined image as a PNG file
                output_image_path = f"/home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/obj/output/combined_image_{epoch}.png"
                cv2.imwrite(output_image_path, combined_image)
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
