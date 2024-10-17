from unicodedata import category
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
import cv2
import os 
import random
import imgviz
import numpy as np
from skimage import measure

def close_contour(contour):
    if not np.array_equal(contour[0],contour[-1]):
        contour=np.vstack((contour,contour[0]))
    return contour
def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons

dataset_dir = 'example/'
json_file = 'new_annotations.json'
coco = COCO(json_file)

categories = coco.loadCats(coco.getCatIds())
names = [category['name'] for category in categories]
# names = [category['supercategory'] for category in categories]

# catIds = coco.getCatIds(catNms=['title'])
# imgIds = coco.getImgIds(catIds=catIds) 

imgids = coco.getImgIds()

for imgid in imgids:
    img_information = coco.loadImgs(imgid)[0]
    if not os.path.exists(dataset_dir + img_information['file_name']):
        continue
    img = cv2.imread(dataset_dir + img_information['file_name'])

    annids = coco.getAnnIds(imgIds=imgid, iscrowd=None)
    anns = coco.loadAnns(annids)
    # coco.showAnns(anns)

    labels = []
    masks = []
    captions = []
    bboxes = []
    for j in range(len(anns)):
        labels.append(anns[j]['category_id'])
        captions.append(names[anns[j]['category_id']-1])

        area = np.array([anns[j]['segmentation']],np.int32).reshape(-1,2)
        mask = cv2.fillPoly(np.zeros_like(img),[area],(1,1,1))[:,:,0]
        # point = binary_mask_to_polygon(mask)
        # point = np.array(point,np.int32).reshape(-1,2)
        # mask = cv2.fillPoly(np.zeros_like(img),[point],(1,1,1))[:,:,0]
        masks.append(mask==1)

        x1 = int(anns[j]['bbox'][0])
        y1 = int(anns[j]['bbox'][1] + anns[j]['bbox'][3])
        x2 = int(anns[j]['bbox'][0] + anns[j]['bbox'][2])
        y2 = int(anns[j]['bbox'][1])
        bboxes.append(np.array([y1,x1,y2,x2]))


    viz = imgviz.instances2rgb(
        image=img,
        labels=labels,
        masks=masks,
        # bboxes=None,
        bboxes=bboxes,
        captions=captions,
        font_size=60,
        line_width=2,
        alpha = 0.3,
        boundary_width=1
    )
    cv2.imwrite('visualize/' + img_information['file_name'],viz)
    # cv2.imshow('viz',cv2.resize(viz,(768,768)))
    # cv2.imshow('viz',viz)
    # cv2.waitKey(0)
