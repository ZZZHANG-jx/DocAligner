from operator import ne
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
import time
from .resampling import rectification
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



class doc3dcrfmsynthLoader(data.Dataset):
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
        im_name = self.files[self.split][index]     # 1/824_8-cp_Page_0503-7Nw0001
        im_path = pjoin(self.root, 'img_dewarp_base_tpsplus',  im_name + '.png')  
        uv_path = pjoin(self.root, 'flow_dewarp_base_tpsplus' , im_name + '.npy')
        pdf_path = pjoin(self.root, 'img_dewarp_base_tpsplus', im_name + '.png')
        uw_im_path = pjoin(self.root, 'img_dewarp_base_tpsplus', im_name + '.png')

        ## prepare im
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im_resize = cv2.resize(im,self.img_size)

        ## prepare pdf
        pdf = cv2.imread(pdf_path)
        pdf = cv2.cvtColor(pdf,cv2.COLOR_BGR2RGB)
        pdf_resize = cv2.resize(pdf,self.img_size)

        ## prepare uw
        uw_im = cv2.imread(uw_im_path)
        uw_im = cv2.cvtColor(uw_im,cv2.COLOR_BGR2RGB)
        uw_im_resize = cv2.resize(uw_im,self.img_size)       

        ## prepare shift
        flow = np.load(uv_path,allow_pickle=True)
        flow = flow.transpose(2,0,1)
        flow_resize = flowResize(flow,self.img_size[0],self.img_size[1])
        mean_x = np.mean(flow_resize[0,:,:].copy())
        mean_y = np.mean(flow_resize[1,:,:].copy())
        flow_resize = flow_resize-np.array([mean_x,mean_y]).reshape(2,1,1)


        ## prepare content mask
        _,content_mask_resize = cv2.threshold(cv2.cvtColor(pdf_resize,cv2.COLOR_RGB2GRAY),240,255,cv2.THRESH_BINARY_INV)

        ## coord_conv
        base_coord = getBasecoord(self.img_size[0],self.img_size[1]) / np.array([self.img_size[0],self.img_size[1]]).reshape(2,1,1)


        if False:
            uw_im_resize1 = dewarp_base_displacement(flow_resize.transpose(1,2,0),im_resize,True)
            cv2.imshow('im',cv2.resize(im_resize,(512,512)))
            cv2.imshow('uw_im',cv2.resize(uw_im_resize,(512,512)))
            cv2.imshow('uw_im1',cv2.resize(uw_im_resize1,(512,512)))
            cv2.waitKey()
            cv2.destroyAllWindows()

        im = self.transform(im_resize).float()
        uw_im = self.transform(uw_im_resize).float()
        pdf = self.transform(pdf_resize).float()
        content_mask = self.transform(content_mask_resize).float()
        base_coord = torch.from_numpy(base_coord).float()
        flow_resize = torch.from_numpy(flow_resize).float()#/2.285
        im = torch.cat((im,base_coord),dim=0)
        # print(flow_resize.min(),flow_resize.max())
        return im, flow_resize, content_mask, uw_im

    def transform(self,img):
        if len(img.shape) == 2 :
            img = np.expand_dims(img,-1)
        img = img.astype(np.float)/255.
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        return img