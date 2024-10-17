import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
import cv2
import random
import hdf5storage as h5
import h5py
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils import data

from .augmentationsk import data_aug, tight_crop

from .dewarp_uv import dewarp_base_displacement
def getBasecoord(h,w):
    base_coord0 = np.tile(np.arange(h).reshape(h,1),(1,w)).astype(np.float32)
    base_coord1 = np.tile(np.arange(w).reshape(1,w),(h,1)).astype(np.float32)
    base_coord = np.concatenate((np.expand_dims(base_coord0,0),np.expand_dims(base_coord1,0)),0)
    return base_coord
def array_normalize(array):
    array = (array-array.min())/(array.max()-array.min())
    return array

def flowResize(flow,target_h,target_w):
    org_h,org_w = flow.shape[1:]
    base_coord = getBasecoord(org_h,org_w)
    new_coord = (flow + base_coord).astype(np.float32)/np.array([org_h,org_w]).reshape(2,1,1)
    resize_new_coord0 = cv2.resize(new_coord[0,:,:],(target_w,target_h))
    resize_new_coord1 = cv2.resize(new_coord[1,:,:],(target_w,target_h))
    resize_new_coord = np.concatenate((np.expand_dims(resize_new_coord0*target_h,0),np.expand_dims(resize_new_coord1*target_w,0)),0)

    base_coord = getBasecoord(target_h,target_w)
    resize_flow = resize_new_coord - base_coord
    return resize_flow

def flowToclassifygt(flow0):
    classify0 = np.clip(flow0.copy(),-100,100)+100
    classify0 = classify0.astype(int)
    # classify0 = classify0.reshape.reshape(-1,200)
    classify0 = np.eye(200)[classify0.reshape(-1)].reshape(-1,200)
    return classify0
def classifygtToflow(classify0):
    classify0 = classify0.reshape(-1,200)
    classify0 = np.argmax(classify0,axis=1).reshape(256,256)
    classify0 -= 100
    return classify0
    

class doc3dcrfmclassifyLoader(data.Dataset):
    """
    Loader for world coordinate regression and RGB images
    """
    def __init__(self, root, split='train', is_transform=False,
                 img_size=512, augmentations=None):
        #self.root = os.path.expanduser(root)
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 3   
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.size = self.img_size[0]

        self.base_coordinate = np.argwhere(np.zeros(self.img_size, dtype=np.uint32) == 0)

        path = pjoin(self.root, split + '.txt')
        file_list = tuple(open(path, 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[split] = file_list
        if self.augmentations:
            self.txpths=[]
            with open(os.path.join(self.root[:-7],'augtexnames.txt'),'r') as f:
                for line in f:
                    txpth=line.strip()
                    self.txpths.append(txpth)


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, 'img_dewarp_base_tpsplus',  im_name + '.png')  
        uv_path = pjoin(self.root, 'uv_dewarp_base_tpsplus' , im_name + '.exr')
        pdf_path = pjoin(self.root, 'alb_dewarp_base_tpsplus', im_name + '.png')

        ## prepare im
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im_resize = cv2.resize(im,self.img_size)
        if not os.path.exists(pdf_path):
            pdf_resize = np.ones_like(im_resize)
        else:
            pdf = cv2.imread(pdf_path)
            pdf = cv2.cvtColor(pdf,cv2.COLOR_BGR2RGB)
            pdf_resize = cv2.resize(pdf,self.img_size)

        # prepare shift
        uv=cv2.imread(uv_path,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        uv_resize = cv2.resize(uv,self.img_size)
        uv_resize[:,:,1] = (np.ones_like(uv_resize[:,:,1]) - uv_resize[:,:,1]).copy()*uv_resize[:,:,0].copy()
        new_coordinates=uv_resize[:,:,1:]
        shift_resize = new_coordinates - self.base_coordinate.reshape(new_coordinates.shape)/np.array([uv_resize.shape[0],uv_resize.shape[1]])
        shift_resize = shift_resize*(uv_resize[:,:,0].reshape(new_coordinates.shape[0],new_coordinates.shape[1],1))
        mean_x = np.mean(shift_resize[:,:,0].copy())
        mean_y = np.mean(shift_resize[:,:,1].copy())
        shift_resize_normalize = shift_resize-np.array([mean_x,mean_y]).reshape(1,1,2)

        ## prepare shift classify
        classify = shift_resize_normalize.copy()*448
        classify0 = cv2.resize(classify[:,:,0],(14,14))
        classify1 = cv2.resize(classify[:,:,1],(14,14))
        classify = np.concatenate((np.expand_dims(classify0,-1),np.expand_dims(classify1,-1)),-1)
        classify = classify.transpose(2,0,1)
        classify = classify+100
        classify = classify.astype(int)
        classify = np.clip(classify,0,199)

        # classify0 = flowToclassifygt(shift_resize_normalize[:,:,0]*256)
        # classify1 = flowToclassifygt(shift_resize_normalize[:,:,1]*256)
        # classify0 = classifygtToflow(classify0)
        # classify1 = classifygtToflow(classify1)
        # reconstruct_shift = np.concatenate((np.expand_dims(classify0,-1),np.expand_dims(classify1,-1)),-1)

        # prepare unwarp image
        uw_im_resize = im_resize

        # prepare content mask
        _,content_mask_resize = cv2.threshold(cv2.cvtColor(pdf_resize,cv2.COLOR_RGB2GRAY),240,255,cv2.THRESH_BINARY_INV)
        content_mask_resize[0:5] = 0
        content_mask_resize[-5:] = 0
        content_mask_resize[:,0:5] = 0
        content_mask_resize[:,-5:] = 0


        if False:
            # uw_im_resize = dewarp_base_displacement(new_coordinates.copy()*im_resize.shape[0],im_resize,False)

            uw_im_resize = dewarp_base_displacement(shift_resize_normalize.copy()*256,im_resize,True)    
            uw_im_resize_reconstruct = dewarp_base_displacement(reconstruct_shift.copy(),im_resize,True)    
            cv2.imshow('im',im_resize)
            cv2.imshow('uw_im',uw_im_resize)
            cv2.imshow('uw_im_reconstruct',uw_im_resize_reconstruct)
            cv2.waitKey(0)


        im = self.transform(im_resize).float()
        uw_im = self.transform(uw_im_resize).float()
        pdf = self.transform(pdf_resize).float()
        content_mask = self.transform(content_mask_resize).float()
        shift = torch.from_numpy(shift_resize_normalize.transpose(2, 0, 1)).float()
        coordinate = self.base_coordinate.reshape(new_coordinates.shape)/np.array([new_coordinates.shape[0],new_coordinates.shape[1]])
        coordinate = torch.from_numpy(coordinate.transpose(2,0,1)).float()
        im = torch.cat((im,coordinate),dim=0)       

        return im, shift*256, content_mask, uw_im,classify

    def transform(self,img):
        if len(img.shape) == 2 :
            img = np.expand_dims(img,-1)
        img = img.astype(np.float)/255.
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        return img