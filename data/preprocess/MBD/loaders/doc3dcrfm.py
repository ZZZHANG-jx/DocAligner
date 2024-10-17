from operator import is_
from loaders.doc3dcrfm_synth import flowResize, getBasecoord
import os
from os.path import join as pjoin
import collections
import json
from numpy.core.numeric import _cross_dispatcher
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

class doc3dcrfmLoader(data.Dataset):
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

        # print(self.img_size)        
        # for split in ['train', 'val','debug']:
        path = pjoin(self.root, split + '.txt')
        file_list = tuple(open(path, 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[split] = file_list
        #self.setup_annotations()
        if self.augmentations:
            self.txpths=[]
            with open(os.path.join(self.root[:-7],'augtexnames.txt'),'r') as f:
                for line in f:
                    txpth=line.strip()
                    self.txpths.append(txpth)


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        is_augment_dewarped = False
        is_augment_crop = False
        is_pdf_exists = True
        im_name = self.files[self.split][index]                # 1/824_8-cp_Page_0503-7Nw0001
        im_path = pjoin(self.root, 'img_dewarp_base_tpsplus',  im_name + '.png')  
        uv_path = pjoin(self.root, 'uv_dewarp_base_tpsplus' , im_name + '.exr')
        pdf_path = pjoin(self.root, 'alb_dewarp_base_tpsplus', im_name + '.png')
        uw_path = pjoin(self.root, 'img_dewarp_base_normalize_uv', im_name + '.png')

        if 'train' in self.split:
            p = 0.7
        if 'val' in self.split:
            p = 1

        if random.uniform(0,1) > p:
            im_path = pjoin(self.root, 'img_dewarp_base_uv',  im_name + '.png')  
            pdf_path = pjoin(self.root, 'alb_dewarp_base_uv', im_name + '.png')
            uw_path = pjoin(self.root, 'img_dewarp_base_uv', im_name + '.png')
            is_augment_dewarped = True
        if random.uniform(0,1) > 1:
            im_path = pjoin(self.root, 'img',  im_name + '.png')  
            pdf_path = pjoin(self.root, 'alb', im_name + '.png')
            uw_path = pjoin(self.root, 'img', im_name + '.png')
            uv_path = pjoin(self.root, 'uv' , im_name + '.exr')
            is_augment_crop = True

        # prepare im
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im_resize = cv2.resize(im,self.img_size)
        if not os.path.exists(pdf_path):
            pdf_resize = np.ones_like(im_resize)
            is_pdf_exists = False
            pdf_resize = im_resize
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
        if is_augment_dewarped == True:
            shift_resize_normalize = np.zeros_like(shift_resize_normalize)     

        ## prepare document mask 
        doc_mask_resize = uv_resize[:,:,0]*255
        # im_resize = im_resize*np.expand_dims(doc_mask_resize,-1)/255

        ## prepare unwarp image
        # uw_im_resize = dewarp_base_displacement(new_coordinates.copy()*im_resize.shape[0],im_resize,False)
        # print(shift_resize.shape,im_resize.shape)
        # uw_im_resize = dewarp_base_displacement(shift_resize_normalize.copy()*448,im_resize,True)       
        uw = cv2.imread(uw_path)
        uw = cv2.cvtColor(uw,cv2.COLOR_BGR2RGB)
        uw_resize = cv2.resize(uw,self.img_size)

        ## prepare content mask
        kernel = np.ones((3,3))
        if is_pdf_exists:
            _,content_mask_resize = cv2.threshold(cv2.cvtColor(pdf_resize,cv2.COLOR_RGB2GRAY),240,255,cv2.THRESH_BINARY_INV)
            content_mask_resize[0:5] = 0
            content_mask_resize[-5:] = 0
            content_mask_resize[:,0:5] = 0
            content_mask_resize[:,-5:] = 0
            content_mask_resize = cv2.dilate(content_mask_resize,kernel,iterations=3)
        else:
            # _,content_mask_resize = cv2.threshold(cv2.cvtColor(pdf_resize,cv2.COLOR_RGB2GRAY),50,255,cv2.THRESH_BINARY_INV)
            content_mask_resize = cv2.adaptiveThreshold(cv2.cvtColor(pdf_resize,cv2.COLOR_RGB2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
            content_mask_resize = cv2.erode(content_mask_resize,kernel)
            content_mask_resize = cv2.dilate(content_mask_resize,kernel,iterations=4)


        ## augmentation
        # im_resize, shift_resize_normalize = self.random_crop(im_resize,shift_resize_normalize)

        if 0:
            flow = shift_resize_normalize.copy()*448
            # flow[np.concatenate((np.expand_dims(doc_mask_resize,-1),np.expand_dims(doc_mask_resize,-1)),-1)==0]=-600
            uw_im_resize = dewarp_base_displacement(flow,im_resize,True)       
            cv2.imshow('im',im_resize)
            # cv2.imshow('uw_im_normalize',uw_im_resize_normalize)
            cv2.imshow('uw_im',uw_im_resize)
            cv2.imshow('pdf',pdf_resize)
            cv2.imshow('mask',content_mask_resize)
            cv2.waitKey(0)


        im = self.transform(im_resize).float()
        uw = self.transform(uw_resize).float()
        pdf = self.transform(pdf_resize).float()
        content_mask = self.transform(content_mask_resize).float()
        doc_mask = self.transform(doc_mask_resize).float()
        shift = torch.from_numpy(shift_resize_normalize.transpose(2, 0, 1)).float()
        coordinate = self.base_coordinate.reshape(new_coordinates.shape)/np.array([new_coordinates.shape[0],new_coordinates.shape[1]])
        coordinate = torch.from_numpy(coordinate.transpose(2,0,1)).float()
        im = torch.cat((im,coordinate),dim=0)  
        # print(shift.max()*448,shift.min()*448)     
        return im, doc_mask, shift*448, content_mask, uw

    def transform(self,img):
        if len(img.shape) == 2 :
            img = np.expand_dims(img,-1)
        img = img.astype(np.float)/255.
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        return img
    
    def random_crop(self,img,shift):
        h,w = img.shape[:2]
        crop_h = 224
        crop_w = 224
        start_h = random.randint(0,h-crop_h)
        start_h = 112
        start_w = random.randint(0,w-crop_w)
        start_w = 112

        new_img = img[start_h:start_h+crop_h,start_w:start_w+crop_w]
        new_img = cv2.resize(new_img,(w,h))
        
        coord = shift + getBasecoord(h,w).transpose(1,2,0)/np.array([h,w]).reshape(1,1,2)
        new_coord = coord[start_h:start_h+crop_h,start_w:start_w+crop_w]
        new_coord[:,:,0] = (new_coord[:,:,0]-new_coord[:,:,0].min())/(new_coord[:,:,0].max()-new_coord[:,:,0].min())
        new_coord[:,:,1] = (new_coord[:,:,1]-new_coord[:,:,1].min())/(new_coord[:,:,1].max()-new_coord[:,:,1].min())

        new_new_coord0 = cv2.resize(new_coord[:,:,0],(w,h))
        new_new_coord1 = cv2.resize(new_coord[:,:,1],(w,h))
        new_new_coord = np.concatenate((np.expand_dims(new_new_coord0,-1),np.expand_dims(new_new_coord1,-1)),-1)

        new_shift = new_new_coord - getBasecoord(h,w).transpose(1,2,0)/np.array([h,w]).reshape(1,1,2)

        new_shift = new_shift - new_shift[int(h/2),int(w/2)]

        return new_img,new_shift

