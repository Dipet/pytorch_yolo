import cv2 as cv
import matplotlib.pyplot as plt

from tqdm import tqdm

import pandas as pd


def show_image(row):
    image = cv.imread(row['image_path'], cv.IMREAD_GRAYSCALE)
    plt.figure(figsize=(15, 15))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    

def show_with_bbox(row):
    left = int(row['left'])
    top = int(row['top'])
    right = int(row['right'])
    bottom = int(row['bottom']) 
    
    image = cv.imread(row['image_path'], cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    
    plt.figure(figsize=(15, 15))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    
def images_to_gray(paths):
    for path in tqdm(paths):
        image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        cv.imwrite(path, image)
        
        
def save_keras_ssd(df: pd.DataFrame, path):
    df.to_csv(path, index=False, columns=['image_path', 'left', 'right', 'top', 'bottom', 'type'], header=False)


def compute_iou(bbox1, bbox2):
    """

    Args:
        predicted_bbox: [left, right, top, bottom]
        true_bbox: [left, right, top, bottom]

    Returns: float [0-1]

    """
    x11, x12, y11, y12 = bbox1
    x21, x22, y21, y22 = bbox2

    assert x11 < x12
    assert y11 < y12
    assert x21 < x22
    assert y21 < y22

    # determine the coordinates of the intersection rectangle
    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (x12 - x11) * (y12 - y11)
    bb2_area = (x22 - x21) * (y22 - y21)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
