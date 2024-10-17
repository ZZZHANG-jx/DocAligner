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

class doc3dcrfmpatchLoader(data.Dataset):
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
        im_name = self.files[self.split][index]                # 1/824_8-cp_Page_0503-7Nw0001
        im_path = pjoin(self.root, 'img_dewarp_base_tpsplus',  im_name + '.png')  
        uv_path = pjoin(self.root, 'uv_dewarp_base_tpsplus' , im_name + '.gw')
        pdf_path = pjoin(self.root, 'alb_dewarp_base_tpsplus', im_name + '.png')
        if not os.path.exists(uv_path):
            print(im_path,'================================')

        crop_size = 448
        ratio = int(self.img_size[0]/crop_size)
        # prepare im
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        crop_y = int(random.uniform(0,im.shape[0]-crop_size))
        crop_x = int(random.uniform(0,im.shape[1]-crop_size))
        im = im[crop_y:crop_y+crop_size,crop_x:crop_x+crop_size,:]
        im_resize = cv2.resize(im,self.img_size)
        if not os.path.exists(pdf_path):
            pdf_resize = np.ones_like(im_resize)
        else:
            pdf = cv2.imread(pdf_path)
            pdf = cv2.cvtColor(pdf,cv2.COLOR_BGR2RGB)
            pdf_resize = cv2.resize(pdf,self.img_size)

        # prepare shift
        uv=cv2.imread(uv_path,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        uv_resize = cv2.resize(uv,(self.img_size[1]*ratio,self.img_size[0]*ratio))
        uv_resize[:,:,1] = (np.ones_like(uv_resize[:,:,1]) - uv_resize[:,:,1]).copy()*uv_resize[:,:,0].copy()
        new_coordinates=uv_resize[:,:,1:]
        self.base_coordinate = np.argwhere(np.zeros((self.img_size[1]*ratio,self.img_size[0]*ratio), dtype=np.uint32) == 0)
        shift_resize = new_coordinates - self.base_coordinate.reshape(new_coordinates.shape)/np.array([uv_resize.shape[0],uv_resize.shape[1]])

        shift_resize = shift_resize*(uv_resize[:,:,0].reshape(new_coordinates.shape[0],new_coordinates.shape[1],1))

        # shift_resize = shift_resize[shift_y:shift_y+int(shift_resize.shape[0]/2),shift_x:shift_x+int(shift_resize.shape[1]/2),:]
        shift_resize = shift_resize[crop_y*ratio:crop_y*ratio+self.img_size[0],crop_x*ratio:crop_x*ratio+self.img_size[1]]
        shift_resize = shift_resize*ratio
        # shift_resize_temp = np.zeros((self.img_size[0],self.img_size[1],2))
        # shift_resize_temp[:,:,0] = cv2.resize(shift_resize[0],self.img_size)
        # shift_resize_temp[:,:,1] = cv2.resize(shift_resize[1],self.img_size)
        # print(shift_resize_temp.max(),shift_resize_temp.min(),shift_resize.max(),shift_resize.min())
        # shift_resize = shift_resize_temp
        # shift_resize = new_coordinates
        


        # prepare unwarp image
        mean_y = np.mean(shift_resize[:,:,0].copy())
        mean_x = np.mean(shift_resize[:,:,1].copy())
        mean_y,mean_x = shift_resize[224,224,:]
        shift_resize_normalize = shift_resize-np.array([mean_y,mean_x]).reshape(1,1,2)
        # uw_im_resize = dewarp_base_displacement(new_coordinates.copy()*im_resize.shape[0],im_resize,False)
        # print(shift_resize.shape,im_resize.shape)
        uw_im_resize = dewarp_base_displacement(shift_resize_normalize.copy()*448,im_resize,True)       
       
       
        # uw_im_resize = im_resize

        # prepare content mask
        # content_mask_resize = pdf_resize
        _,content_mask_resize = cv2.threshold(cv2.cvtColor(pdf_resize,cv2.COLOR_RGB2GRAY),240,255,cv2.THRESH_BINARY_INV)
        content_mask_resize[0:5] = 0
        content_mask_resize[-5:] = 0
        content_mask_resize[:,0:5] = 0
        content_mask_resize[:,-5:] = 0


        if True:
            cv2.imshow('im',im_resize)
            # cv2.imshow('uw_im_normalize',uw_im_resize_normalize)
            cv2.imshow('uw_im',uw_im_resize)
            # cv2.imshow('pdf',pdf_resize)
            # cv2.imshow('mask',content_mask_resize)
            cv2.waitKey(0)
#        if 'val' in self.split:
#            im, depth=tight_crop(im/255.0,depth)
#            im, wc=tight_crop(im/255.0,wc)
#        if self.augmentations:          #this is for training, default false for validation\
#            tex_id=random.randint(0,len(self.txpths)-1)
#            txpth=self.txpths[tex_id] 
#            tex=cv2.imread(os.path.join(self.root[:-7],txpth)).astype(np.uint8)
#            bg=cv2.resize(tex,self.img_size,interpolation=cv2.INTER_NEAREST)
#            im,depth,msk=data_aug(im,depth,bg)

        im = self.transform(im_resize).float()
        # gt = self.transform(gt)
        uw_im = self.transform(uw_im_resize).float()
        pdf = self.transform(pdf_resize).float()
        content_mask = self.transform(content_mask_resize).float()
        shift = torch.from_numpy(shift_resize.transpose(2, 0, 1)).float()
        shift = torch.from_numpy(shift_resize.transpose(2, 0, 1)).float()
        self.base_coordinate = np.argwhere(np.zeros(self.img_size, dtype=np.uint32) == 0)
        coordinate = self.base_coordinate.reshape(448,448,2)/np.array([448,448])
        coordinate = torch.from_numpy(coordinate.transpose(2,0,1)).float()
        im = torch.cat((im,coordinate),dim=0)       
        # if self.is_transform:
        #     im_org, im, depth_org,depth, mask, edge, bm_org, bm,content_mask_org = self.transform(im, depth, mask, edge, bm,content_mask)
        # print(shift.max(),shift.min())
        return im, shift, content_mask, uw_im

    def transform(self,img):
        if len(img.shape) == 2 :
            img = np.expand_dims(img,-1)
        img = img.astype(np.float)/255.
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        return img

    # def transform(self, img, depth, mask, edge,bm,content_mask):
    #     # img 
    #     # if img.shape[-1] == 4:
    #         # img=img[:,:,:3]   # Discard the alpha channel  
    #     # img = img[:, :, ::-1] # RGB -> BGR
    #     # img = img.transpose(2, 0, 1) # NHWC -> NCHW

    #     #depth
    #     xmx, xmn, ymx, ymn,zmx, zmn= 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497   # calculate from all the wcs
    #     depth = (depth-zmn)/(zmx-zmn)
    #     minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(depth, mask)
    #     depth = (depth-minVal)/(maxVal-minVal)*(mask/255)
    #     # depth_org = depth
    #     # depth = cv2.resize(depth, self.img_size)

    #     # mask
    #     mask = mask.astype(float)/255.

    #     # edge
    #     edge = edge.astype(float)/255.
    #     edge[edge > 0.1] = 1.
    #     edge[edge <= 0.1] = 0.

    #     bm = bm.astype(float)
    #     # print(bm.max(),bm.min())
    #     bm=bm/448.0

    #     #content_mask
    #     content_mask = content_mask.astype(float)/255
    #     content_mask[content_mask > 0.1] =1.
    #     content_mask[content_mask <= 0.1 ] =0.



    #     # to torch
    #     img = img.astype(float) / 255.0
    #     img = img[:, :, ::-1]
    #     img_org = img.copy()
    #     img = cv2.resize(img, self.img_size)
    #     img_org = img_org.transpose((2,0,1))
    #     img = img.transpose((2,0,1))
    #     img = torch.from_numpy(img).float()
    #     img_org = torch.from_numpy(img_org).float()
    #     # print(img.shape,'img')
    #     # print(self.img_size,type(self.img_size))

    #     depth_org = depth.copy()
    #     depth = cv2.resize(depth,self.img_size)
    #     depth = torch.from_numpy(depth).float()
    #     depth_org = torch.from_numpy(depth_org).float()
    #     depth = torch.unsqueeze(depth,0)
    #     depth_org = torch.unsqueeze(depth_org,0)
    #     # print(depth.shape,'depth')

    #     mask = cv2.resize(mask,self.img_size)
    #     mask = torch.from_numpy(mask).float()
    #     mask = torch.unsqueeze(mask,0)
    #     # print(mask.shape,'mask')
    #     # mask = F.interpolate(mask, self.img_size)

    #     edge = cv2.resize(edge,self.img_size)
    #     edge = torch.from_numpy(edge).float()
    #     edge = torch.unsqueeze(edge,0)
    #     # print(edge.shape,'edge')
    #     # edge = F.interpolate(edge, self.img_size)

    #     content_mask = torch.from_numpy(content_mask).float()
    #     content_mask = torch.unsqueeze(content_mask,0)

    #     # bm = bm.transpose((1,2,0))
    #     bm_org = bm.copy()
    #     bm = cv2.resize(bm,self.img_size)
    #     bm_org = cv2.resize(bm_org,self.img_size)
    #     bm = bm.transpose((2,0,1))
    #     bm = torch.from_numpy(bm).float()
    #     bm_org = bm_org.transpose((2,0,1))
    #     bm_org = torch.from_numpy(bm_org).float()
    #     # print(bm.shape,'bm')
    #     # print(img_org.shape,img.shape,depth_org.shape,depth.shape,mask.shape,edge.shape,bm_org.shape,bm.shape)
    #     return img_org,img,depth_org,depth,mask,edge,bm_org,bm,content_mask

