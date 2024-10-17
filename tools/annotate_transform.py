from hashlib import new
import os 
import numpy as np 
import glob 
import cv2 
import json 
import  torch.nn.functional as F
import torch
from pycocotools import mask as maskUtils
from skimage import measure
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from tqdm import tqdm

def segpointsToMask(segm, img_size):
    h, w = img_size
    rles = maskUtils.frPyObjects(segm, h, w)
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)
    m = (m*255).astype(np.uint8)
    return m


def maskToSegpoints(binary_mask):
    contour = measure.find_contours(binary_mask)[0]
    contour = np.flip(contour, axis=1)
    segmentation = contour.ravel().tolist()
    return segmentation

def sparse_sampling(segmentation):
    ratio = 3
    num = int(len(segmentation)/2)
    new_segmentation = []
    for i in range(0,num,2*ratio):
        new_segmentation.append(segmentation[i],segmentation[i+1])
    return new_segmentation


def point_transform_batch_accelorate_cuda_fm(points_list,grid,org_h,org_w):
    
    ## preprocess 
    points_num = int(len(points_list)/2)
    new_points_list = []
    for i in range(points_num):
        new_points_list.append((points_list[i*2],points_list[i*2+1]))
    points_list = new_points_list

    size = 512
    '''
    forward dewarping 
    input:
        points_list -> [(w0,h0),(w1,h1)...], a list consists of absolute coordinates
        new_coordinate -> torch.Tensor, (1,h,w,2)
    outputL
        new_points -> [(w0,h0),(w1,h1)...], a list consists of new absolute coordinates
    '''
    points = np.asarray(points_list).reshape(-1,2)
    points = points/np.asarray((org_w,org_h))*np.asarray((size,size))
    points = points.astype(int)
    points_num = points.shape[0]
    points = torch.from_numpy(points).cuda()

    new_coordinate = F.interpolate(grid.permute(0,3,1,2),(size,size),mode='bilinear')
    new_coordinate = new_coordinate.permute(0,2,3,1)[0]
    new_coordinate = (new_coordinate+1)/2*(torch.Tensor((org_w,org_h)).view(-1,2).cuda())

    new_points_array = new_coordinate[points[:,1],points[:,0]]
    new_points_array = new_points_array.cpu().numpy().astype(int)
    new_points = [(new_point[0],new_point[1]) for new_point in new_points_array]

    ## postprocess 
    new_new_points = []
    for point in new_points:
        new_new_points.append(float(point[0]))
        new_new_points.append(float(point[1]))
    new_points = new_new_points

    return new_points


grid3_paths = glob.glob('./example/*_grid3*')
json_path = './annotations.json'
coco = COCO(json_path)

with open(json_path,'r') as f:
    lines = f.readlines()
    datas = json.loads(lines[0])



new_datas = {'images':[],'annotations':[],'categories':datas['categories']}
for id, grid3_path in tqdm(enumerate(grid3_paths)):
    target_path = grid3_path.replace('_grid3.jpg.npy','_target.jpg')
    target = cv2.imread(target_path)
    origin_path = grid3_path.replace('_grid3.jpg.npy','_origin.jpg')
    origin = cv2.imread(origin_path)
    org_h,org_w = origin.shape[:2]
    grid3 = np.load(grid3_path)
    grid3 = torch.from_numpy(grid3).unsqueeze(0).cuda()
    file_name = os.path.split(grid3_path)[-1].replace('_grid3.jpg.npy','_target.jpg')

    ## load 
    images = datas['images']
    annotations = datas['annotations']

    ## find the image id
    for image in images:
        if image['file_name'] == file_name:
            id = image['id']
            width = image['width']
            height = image['height']
    ## add new image 
    new_datas['images'].append({'id': id, 'file_name': os.path.split(origin_path)[-1], 'width': org_w, 'height': org_h})


    ## get annotations
    annids = coco.getAnnIds(imgIds=[id], iscrowd=None)
    anns = coco.loadAnns(annids)

    ## change annotations
    for ann in anns:
        segmentation = ann['segmentation']
        bbox = ann['bbox']
        area = ann['area']
        image_id = ann['image_id']

        ## update segmentation
        mask = segpointsToMask(segmentation,(height,width))
        mask = cv2.resize(mask,(org_w,org_h))
        new_segmentation = maskToSegpoints(mask==255)
        new_segmentation = point_transform_batch_accelorate_cuda_fm(new_segmentation,grid3,org_h,org_w)
        new_segmentation = [new_segmentation]
        ann['segmentation'] = new_segmentation
    
        ## update image id 
        ann['image_id'] = id

        ## update bbox & area
        new_mask = segpointsToMask(new_segmentation,(org_h,org_w))
        fortran_new_mask = np.asfortranarray(new_mask==255)
        encoded_new_mask = maskUtils.encode(fortran_new_mask)
        new_area = maskUtils.area(encoded_new_mask)
        new_area = int(new_area)
        new_bbox = maskUtils.toBbox(encoded_new_mask).tolist()
        ann['bbox'] = new_bbox
        ann['area'] = new_area
        
        ## add new annotation
        new_datas['annotations'].append(ann)

        # plt.imshow(origin) 
        # coco.showAnns([ann])
        # plt.show()

with open('new_annotations.json','w') as f1:
    string = json.dumps(new_datas,ensure_ascii=False)
    f1.write(string)

