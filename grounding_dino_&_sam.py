# -*- coding: utf-8 -*-
"""Grounding DINO & SAM
# %cd {HOME}/GroundingDINO

!git clone https://github.com/IDEA-Research/GroundingDINO.git
# %cd {HOME}/GroundingDINO
!git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
!pip install -q -e .

!pip install 'git+https://github.com/facebookresearch/segment-anything.git'

!pip install supervision==0.12.0

# %cd {HOME}
!mkdir -p {HOME}/weights
# %cd {HOME}/weights
!wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
"""
import torch
import os
import cv2
import json
import supervision as sv
from groundingdino.util.inference import Model

print(torch.__version__)
print(os.getcwd())

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

GROUNDING_DINO_CONFIG_PATH = '/home/umit/Downloads/GD/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
GROUNDING_DINO_CHECKPOINT_PATH = '/home/umit/Downloads/GD/weights/groundingdino_swint_ogc.pth'

GD_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=DEVICE)

CLASSES = ['stoplight', 'sign']

video_path = "video_radar.mp4"
cap = cv2.VideoCapture(video_path)
save_folder = "res_" + os.path.basename(video_path)
os.makedirs(save_folder, exist_ok=True)
json_path = os.path.join(save_folder, "predictions.json")

predictions = []
stoplight_id_counter = 0
sign_id_counter = 0

def write_to_json(predictions_boxes, frame_id, json_path):
    data = {
        "frame_id": frame_id,
        "stoplight_boxes": [],
        "sign_boxes": []
    }

    global stoplight_id_counter
    global sign_id_counter

    for box, class_id in zip(predictions_boxes.xyxy, predictions_boxes.class_id):
        if class_id is None:
            continue
        if CLASSES[class_id] == 'stoplight':
            data['stoplight_boxes'].append({"box": box.tolist(), "id": stoplight_id_counter, "status": ""})
            stoplight_id_counter += 1
        elif CLASSES[class_id] == 'sign':
            data['sign_boxes'].append({"box": box.tolist(), "id": sign_id_counter, "description": ""})
            sign_id_counter += 1

    predictions.append(data)
    with open(json_path, 'w') as f:
        json.dump(predictions, f, indent=4)

frame_id = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    if frame_id % 10 == 0:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = GD_model.predict_with_classes(
            image=image_rgb,
            classes=CLASSES,
            box_threshold=0.3,
            text_threshold=0.3
        )

        if len(detections.class_id) == 0 or all(class_id is None for class_id in detections.class_id):
            print(f"No detections in frame {frame_id}")
            frame_id += 1
            continue

        box_annotator = sv.BoxAnnotator()
        labels = []
        for class_id in detections.class_id:
            if class_id is not None:
                if CLASSES[class_id] == 'stoplight':
                    labels.append(f'sftoplight {stoplight_id_counter}')
                    stoplight_id_counter += 1
                elif CLASSES[class_id] == "sign":
                    labels.append(f'sign {sign_id_counter}')
                    sign_id_counter += 1

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections, skip_label=False, labels=labels)

        frame_save_path = os.path.join(save_folder, f"frame_{frame_id}.png")
        cv2.imwrite(frame_save_path, annotated_frame)

        write_to_json(detections, frame_id, json_path)

    frame_id += 1

cap.release()
print(f"Results saved to {json_path} and {save_folder}.")


# import cv2
# image_bgr = cv2.imread('/content/1.png')
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# image_original = image_bgr

# print(image_bgr.shape)

# image_bgr = cv2.resize(image_bgr, (1024, 1024))
# image_rgb = cv2.resize(image_rgb, (1024, 1024))
# image_original = cv2.resize(image_original, (1024, 1024))

# print(image_bgr.shape)

# import supervision as sv
# sv.plot_image(image_bgr)

# # detect objects
# CLASSES = ['license plate', 'tail light']
# detections = GD_model.predict_with_classes(
#     image=image_rgb,
#     classes=CLASSES,
#     box_threshold=0.35,
#     text_threshold=0.1,
# )

# print(detections)

# print(detections.xyxy, type(detections.xyxy))
# detected_boxes = detections.xyxy
# class_id = detections.class_id
# print(class_id)

# import supervision as sv
# box_annotator = sv.BoxAnnotator()
# annotated_frame = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=False, labels=[class_id])
# sv.plot_image(annotated_frame)



# import cv2
# import os
# video_path = "2024-04-26T13_07_35_front.avi"
# cap = cv2.VideoCapture(video_path)

# save_folder = "detect_pred_" + os.path.basename(video_path)
# json_path = os.path.join(save_folder, "predictions.json")

# """
# [
# {frame_id: 1,
# car_boxes: [Ccoords1],
# lp_boxes: [Lcoords1, Lcoords2],
# headlight_boxes: [Hcoords1, Hcoords2]
# }

# {frame_id: 2,
# car_boxes: [Ccoords1],
# lp_boxes: [Lcoords1, Lcoords2],
# headlight_boxes: [Hcoords1]
# }
# ]
# """

# while cap.isOpened():

#     print("frame: ", istart)
#     # Read a frame from the video
#     success, frame = cap.read()
#     if success:
#       cur_frame = cap.get(1)
#       if cur_frame%10 == 0:
#         predictions_boxes = model.inference(frame)
#         if len(predictions_boxes) > 0:
#           curr_frame_save_path = os.path.join(
#               save_folder, cur_frame + ".png"
#           )
#           cv2.saveframe(curr_frame_save_path, frame)
#           write_to_json(predictions_boxes, cur_frame, json_path)

# MODEL_TYPE = "vit_h"
# CHECKPOINT_PATH = '/content/weights/sam_vit_h_4b8939.pth'

# from segment_anything import sam_model_registry, SamPredictor

# sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

# mask_predictor = SamPredictor(sam)

# import numpy as np
# mask_annotator =  sv.MaskAnnotator(color=sv.Color.blue())
# segmented_mask = []
# counter = 0
# for mybox in detected_boxes:
#     mybox = np.array(mybox)
#     print(mybox)


#     mask_predictor.set_image(image_rgb)
#     masks, scores, logits = mask_predictor.predict(
#         point_coords=None,
#         point_labels=None,
#         box=mybox,
#         multimask_output=False
#     )

#     segmented_mask.append(masks)
#     print(len(masks), masks.shape)

# # plot mask on image using supervision
#     detections = sv.Detections(
#         xyxy=sv.mask_to_xyxy(masks=masks),
#         mask=masks
#     )

#     detections = detections[detections.area == np.max(detections.area)]
#     print(CLASSES[class_id[counter]])

#     annotated_image = box_annotator.annotate(scene=image_original.copy(), detections=detections, skip_label=False, labels=[CLASSES[class_id[counter]]])
#     annotated_image = mask_annotator.annotate(scene=annotated_image.copy(), detections=detections)
#     image_original = annotated_image

#     counter+=1

# """Plot image using Supervion"""

# sv.plot_images_grid(
#     images=[image_bgr, annotated_image],
#     grid_size=(1,2),
#     titles=['Original Image', 'Mask Image']

# )

# print(len(segmented_mask), type(segmented_mask[0]), segmented_mask[0].shape)

# for i in range(len(segmented_mask)):

#   segmented_mask[i] = segmented_mask[i].transpose(1,2,0)
#   segmented_mask[i] = np.array(segmented_mask[i]*255).astype('uint8')
#   segmented_mask[i]  = cv2.cvtColor(segmented_mask[i] , cv2.COLOR_GRAY2BGR)


# print(segmented_mask[0].shape)

# sv.plot_images_grid(
#     images=segmented_mask,
#     grid_size=(1, len(segmented_mask)),

# )

# segmented_image = segmented_mask[0]

# for i in range(len(segmented_mask)):
#   try:
#     segmented_image = cv2.bitwise_or(segmented_image, segmented_mask[i+1])
#   except:
#     pass

# sv.plot_image(segmented_image)

# segmented_image = cv2.bitwise_and(segmented_image, image_bgr)
# sv.plot_image(segmented_image)

# segmented_image[np.where((segmented_image == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
# sv.plot_image(segmented_image)

# sv.plot_images_grid(
#     images=[image_bgr, annotated_image, segmented_image],
#     grid_size=(1, 3),
#     titles=['Original Image', 'Annotated Image', 'Segmented Image'],
#     #size=(48,48)

# )

