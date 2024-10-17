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
import hdf5storage as h5
import h5py
import cv2
import random

from tqdm import tqdm
from torch.utils import data

from .augmentationsk import data_aug, tight_crop

class doc3dgierealLoader(data.Dataset):
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
        ## prepare im
        im_name = self.files[self.split][index]                # 1/824_8-cp_Page_0503-7Nw0001
        im_path = pjoin(self.root, 'train_real/image',  im_name) 
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im_resize = cv2.resize(im,self.img_size)  # -> HWC ndarray uint8

        ## prepare mask
        # lbl_path = pjoin(self.root, 'wc', im_name + '.exr')
        # lbl = cv2.imread(lbl_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        # mask = ((lbl[:,:,0]!=0)&(lbl[:,:,1]!=0)&(lbl[:,:,2]!=0)).astype(np.uint8)*255 
        mask = cv2.imread(pjoin(self.root, 'train_real/mask', im_name.replace('_in','_mask')))[:,:,0]
        mask_resize = cv2.resize(mask,self.img_size)    # -> HW ndarray uint8
        
        ## prepare edge
        mask_blur = cv2.blur(mask_resize,(3,3))
        kernel = np.ones((3,3))
        edge = cv2.Canny(mask_blur,20,150)
        edge = cv2.dilate(edge, kernel=kernel)
        edge_resize = cv2.resize(edge, self.img_size)
        edge_resize[edge_resize > 10] = 255
        edge_resize[edge_resize <= 10] = 0  # -> HW ndarray uint8

        ## prepare depth
        # depth = cv2.split(lbl)[0]
        # depth = np.array(depth, dtype=np.float)
        # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(depth, mask)
        # depth = (depth-minVal)/(maxVal-minVal)*(mask)
        # depth = depth.astype(np.uint8)
        # depth_resize = cv2.resize(depth,self.img_size)

        ## augmentation
        if 'train' in self.split or 'debug' in self.split:
            ### add background
            back_ground_path = random.choice(glob.glob('/media/jiaxin/learning_data/dewarp/dataset_render/displacement/background/*.jpg'))
            # cv2.imshow('back',back_ground)
            # cv2.waitKey(0)
            if random.uniform(0,1) > 0.3:
                # back_ground = cv2.imread(back_ground_path)
                im_resize,mask_resize,edge_resize = self.random_augment(im_resize,mask_resize,edge_resize)
            input_edge_resize = cv2.Laplacian(im_resize,cv2.CV_32F)
            input_edge_resize = cv2.convertScaleAbs(input_edge_resize)
            im_blur = cv2.GaussianBlur(im_resize,(15,15),0,0)
            im_resize = im_blur

            ### erase augment
            im_resize = self.erase_augment(im_resize)         # -> HWC ndarray uint8





#        if 'val' in self.split:
#            im, depth=tight_crop(im/255.0,depth)
#            im, lbl=tight_crop(im/255.0,lbl)
#        if self.augmentations:          #this is for training, default false for validation\
#            tex_id=random.randint(0,len(self.txpths)-1)
#            txpth=self.txpths[tex_id] 
#            tex=cv2.imread(os.path.join(self.root[:-7],txpth)).astype(np.uint8)
#            bg=cv2.resize(tex,self.img_size,interpolation=cv2.INTER_NEAREST)
#            im,depth,msk=data_aug(im,depth,bg)
        # im, depth_org, depth, mask, edge = self.transform(im, depth, mask, edge)

        if 0:
            cv2.imshow('im',im_resize)
            cv2.imshow('mask',mask_resize)
            cv2.imshow('edge',edge_resize)
            cv2.imshow('input_edge',input_edge_resize)
            cv2.waitKey(0)


        im = self.transform(im_resize).float()
        mask = self.transform(mask_resize).float()
        edge = self.transform(edge_resize).float()
        # print(edge.shape,mask.shape,im.shape)
        # depth_org = self.transform(depth).float()
        # depth = self.transform(depth_resize).float()
        # for i in [im,mask,edge,depth_org,depth]:
        #     print(i.dtype)
        # exit()
        return im, mask, edge


    def transform(self,img):
        if len(img.shape) == 2 :
            img = np.expand_dims(img,-1)
        img = img.astype(np.float)/255.
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        return img

    def random_scale(self,img,mask,probability=0.5,angle1=-45,angle2=45,scale1=0.8,scale2=1.5):
        y,x = img.shape[:2]
        if random.uniform(0,1) <= probability:
            angle = random.uniform(low=angle1,high=angle2)
            scale = random.uniform(low=scale1,high=scale2)
            M = cv2.getRotationMatrix2D((int(x/2),int(y/2)),angle,scale)
            img_out = cv2.warpAffine(img,M,(x,y))
            mask_out = cv2.warpAffine(mask,M(x,y))
        return img_out,mask_out


    def erase_augment(self,img,probability=0.5):
        if np.random.uniform(0,1) > probability:
            return img
        else:
            h,w = img.shape[:2]
            area = int(np.random.uniform(0.01,0.05)*h*w)
            ration = np.random.uniform(0.5,2)
            h_shift = int(np.sqrt(area*ration))
            w_shift = int(np.sqrt(area/ration))
            h_start = np.random.randint(0,h-h_shift)
            w_start = np.random.randint(0,w-w_shift)
            randm_area = np.random.randint(low=0,high=255,size=(h_shift,w_shift,3))
            img[h_start:h_start+h_shift,w_start:w_start+w_shift,:] = randm_area
            return img

    def randmCrop_augment(self,img,mask,probability=0.5):
        temp1 = np.where(np.sum(mask,axis=0)!=0).nonzero()
        min_w,max_w = temp1[0][0],temp1[0][-1]

        temp2 = np.where(np.sum(mask,axis=0)!=1).nonzero()
        min_h,max_h = temp1[0][0],temp1[0][-1]
    def random_augment(self,img,mask,edge):
        ## random crop
        # low = 0.2
        # high =1.0
        # ratio = np.random.uniform(low,high)
        # crop_size = int(min(back_ground.shape[0],back_ground.shape[1])*ratio)
        # shift_y = np.random.randint(0,back_ground.shape[1]-crop_size)
        # shift_x = np.random.randint(0,back_ground.shape[0]-crop_size)
        # crop_back_ground = back_ground[shift_x:shift_x+crop_size,shift_y:shift_y+crop_size,:]
        # crop_back_ground = cv2.resize(crop_back_ground,self.img_size)
        ## brightness
        high = 1.3
        low = 0.5
        ratio = np.random.uniform(low,high)
        img = img.astype(np.float64)*ratio
        img = np.clip(img,0,255).astype(np.uint8)
        ## contrast
        high = 1.3
        low = 0.5
        ratio = np.random.uniform(low,high)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        mean_array = np.ones_like(img).astype(np.float64)*mean
        img = img.astype(np.float64)*ratio + mean_array*(1-ratio)
        img = np.clip(img,0,255).astype(np.uint8)
        ## color
        high = 0.3
        low = 0.1
        ratio = np.random.uniform(0.1,0.3)
        random_color = np.random.randint(50,200,3).reshape(1,1,3)
        random_color = (random_color*ratio).astype(np.uint8)
        random_color = np.tile(random_color,(self.img_size[0],self.img_size[1],1))
        img = img.astype(np.float64)*(1-ratio) + random_color
        img = np.clip(img,0,255).astype(np.uint8)        
        ## 
        if random.uniform(0,1) > 0.5:
            img = cv2.flip(img,0)
            mask = cv2.flip(mask,0)
            edge = cv2.flip(edge,0)
        if random.uniform(0,1) > 0.5:
            img = cv2.flip(img,1)
            mask = cv2.flip(mask,1)
            edge = cv2.flip(edge,1)
        # y,x = self.img_size
        # angle = random.uniform(-180,180)
        # scale = random.uniform(0.4,1.2)
        # M = cv2.getRotationMatrix2D((int(x/2),int(y/2)),angle,scale)
        # img = cv2.warpAffine(img,M,(x,y),borderValue=0)
        # mask = cv2.warpAffine(mask,M,(x,y),borderValue=0)
        # edge = cv2.warpAffine(edge,M,(x,y),borderValue=0)

        return img,mask,edge






    def random_back_ground(self,back_ground,img,mask,edge_resize):
        ## random crop
        low = 0.2
        high =1.0
        ratio = np.random.uniform(low,high)
        crop_size = int(min(back_ground.shape[0],back_ground.shape[1])*ratio)
        shift_y = np.random.randint(0,back_ground.shape[1]-crop_size)
        shift_x = np.random.randint(0,back_ground.shape[0]-crop_size)
        crop_back_ground = back_ground[shift_x:shift_x+crop_size,shift_y:shift_y+crop_size,:]
        crop_back_ground = cv2.resize(crop_back_ground,self.img_size)
        ## brightness
        high = 1.3
        low = 0.5
        ratio = np.random.uniform(low,high)
        crop_back_ground = crop_back_ground.astype(np.float64)*ratio
        crop_back_ground = np.clip(crop_back_ground,0,255).astype(np.uint8)
        ## contrast
        high = 1.3
        low = 0.5
        ratio = np.random.uniform(low,high)
        gray = cv2.cvtColor(crop_back_ground,cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        mean_array = np.ones_like(crop_back_ground).astype(np.float64)*mean
        crop_back_ground = crop_back_ground.astype(np.float64)*ratio + mean_array*(1-ratio)
        crop_back_ground = np.clip(crop_back_ground,0,255).astype(np.uint8)
        ## color
        high = 0.3
        low = 0.1
        ratio = np.random.uniform(0.1,0.3)
        random_color = np.random.randint(50,200,3).reshape(1,1,3)
        random_color = (random_color*ratio).astype(np.uint8)
        random_color = np.tile(random_color,(self.img_size[0],self.img_size[1],1))
        crop_back_ground = crop_back_ground.astype(np.float64)*(1-ratio) + random_color
        crop_back_ground = np.clip(crop_back_ground,0,255).astype(np.uint8)        
        ## 
        y,x = self.img_size
        angle = random.uniform(-180,180)
        scale = random.uniform(0.5,1.5)
        M = cv2.getRotationMatrix2D((int(x/2),int(y/2)),angle,scale)
        img = cv2.warpAffine(img,M,(x,y),borderValue=0)
        mask = cv2.warpAffine(mask,M,(x,y),borderValue=0)
        edge_resize = cv2.warpAffine(edge_resize,M,(x,y),borderValue=0)

        result_img_flat = crop_back_ground.copy().reshape(self.img_size[0]*self.img_size[1]*3)
        mask_flat = np.tile(np.expand_dims(mask,-1),(1,1,3)).reshape(self.img_size[0]*self.img_size[1]*3)
        img_flat = img.copy().reshape(self.img_size[0]*self.img_size[1]*3)
        result_img_flat[mask_flat>0] = img_flat[mask_flat>0]
        result_img = result_img_flat.reshape(self.img_size[0],self.img_size[1],3)
        return result_img,mask,edge_resize
