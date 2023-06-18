import os
import json
import cv2
import numpy as np
from glob import glob

def get_bounding_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    return bounding_boxes

def convert_dataset(images_dir, masks_dir, output_json):
    image_files = sorted(glob(os.path.join(images_dir, '*.png')))
    mask_files = sorted(glob(os.path.join(masks_dir, '*.png')))

    images = []
    annotations = []

    ann_id = 1
    for i, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
        image = cv2.imread(image_file)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        height, width = image.shape[:2]
        file_name = os.path.basename(image_file)

        images.append({
            'file_name': file_name,
            'height': height,
            'width': width,
            'id': i+1
        })

        bounding_boxes = get_bounding_box(mask)
        for bbox in bounding_boxes:
            x, y, w, h = bbox
            annotations.append({
                'bbox': [x, y, w, h],
                'category_id': 1,
                'image_id': i+1,
                'id': ann_id,
                'iscrowd': 0,
                'area': w * h
            })
            ann_id += 1

    coco_format = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': 1, 'name': 'foreground'}]
    }

    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)

if __name__ == '__main__':
    images_dir = "/home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split4951/fold1/P49/imgs"  # dir containing input images
    masks_dir = "/home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split4951/fold1/P49/labels"  # dir containing input masks
    output_dir = "/home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split4951/fold1/P49"  #
    # images_dir = 'path/to/images'
    # masks_dir = 'path/to/masks'
    output_json = os.path.join(output_dir,'annotations.json')

    convert_dataset(images_dir, masks_dir, output_json)



##visualize
# import os
# import cv2
# import json
# import numpy as np
#
# json_file = "/home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split4951/fold1/P51/instances_pranet.json"
# image_dir = "/home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split4951/fold1/P51/imgs"  # dir containing input images
# mask_dir = "/home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split4951/fold1/P51/labels"  # dir containing input masks
# output_dir = 'temp'
# os.makedirs(output_dir, exist_ok=True)
#
# # Load COCO JSON
# with open(json_file) as f:
#     coco_data = json.load(f)
#
# # Get image info and annotations
# image_infos = coco_data["images"]
# annotations = coco_data["annotations"]
#
# # Process images
# for image_info in image_infos:
#
#     # Load mask
#     mask_path = os.path.join(mask_dir, image_info["file_name"])
#     mask = cv2.imread(mask_path, 0)
#
#     # Get bounding boxes for this image
#     image_id = image_info["id"]
#     bboxes = []
#     for annotation in annotations:
#         if annotation["image_id"] == image_id:
#             bboxes.append(annotation["bbox"])
#
#     # Draw bounding boxes on mask
#     for bbox in bboxes:
#         cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), 255, 2)
#
#         # Save output
#     output_path = os.path.join(output_dir, image_info["file_name"])
#     cv2.imwrite(output_path, mask)
#
# print("Done! Output masks saved to {}".format(output_dir))