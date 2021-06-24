import cv2
import numpy as np
import torch
import os

# crop face from bbox


def crop_face_from_frame_with_bbox_wav2lip(frame, bbox):

    output_size = 512

    pady1, pady2, padx1, padx2 = [0, 0, 0, 0]

    y1 = int(round(max(0, bbox[1] - pady1)))
    y2 = int(round(min(frame.shape[0], bbox[3] + pady2)))
    x1 = int(round(max(0, bbox[0] - padx1)))
    x2 = int(round(min(frame.shape[1], bbox[2] + padx2)))

    crop_img = frame[y1:y2, x1:x2]
    return crop_img

# comput aspect preserved bbox


def compute_aspect_preserved_bbox(bbox, increase_area):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top

    width_increase = max(
        increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(
        increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    return (left, top, right, bot)

# check of bbox-A is inside bbox-B


def one_box_inside_other(boxA, boxB):
    xA = boxA[0] <= boxB[0]
    yA = boxA[1] <= boxB[1]
    xB = boxA[2] >= boxB[2]
    yB = boxA[3] >= boxB[3]
    return xA and yA and xB and yB

# calculate bbox from 68 points facial landmarks


def get_bounding_box_from_landmarks(pts):
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    return (min_x, min_y, max_x, max_y)


# check wether two bounding box intersect over union
def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# check camera motion
def check_camera_motion(current_frame, previous_frame):
    flow = cv2.calcOpticalFlowFarneback(
        previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.quantile(mag, [0.25, 0.5, 0.75], overwrite_input=True)

# get smooth bbox


def get_smoothened_boxes(boxes, T):
    difference = (T - 1)//2

    new_boxes = []
    for i in range(len(boxes)):
        
        if(i == 0):
            window = [boxes[i]]
            # new_boxes.append(window)
        elif (i == 1):
            window = boxes[(i - 1):(i + 1)]
            # new_boxes.append(np.mean(window, axis=0))
        elif (i == len(boxes)-1):
            window = [boxes[i]]
            # new_boxes.append(window)
        elif (i == len(boxes)-2):
            window = boxes[(i - 1):(i + 1)]
            # new_boxes.append(np.mean(window, axis=0))
        else:
            window = boxes[(i - difference):(i + difference)]
            # new_boxes.append(np.mean(window, axis=0))
        
        new_boxes.append(np.mean(window, axis=0))
    return new_boxes