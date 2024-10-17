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

class doc3dbmpLoader(data.Dataset):
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
        # print(self.img_size)        
        for split in ['train', 'val','debug']:
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
        im_path = pjoin(self.root, 'img',  im_name + '.png')  
        bm_path = pjoin(self.root, 'bm_npy' , im_name + '.npy')
        pdf_path = pjoin(self.root, 'alb', im_name + '.png')

        # prepare bm
        bm = np.load(bm_path) # -> ndarray (448*448*2) 0~448 float

        # prepare sm
        base = np.arange(448)
        base_x = np.expand_dims(a,0).repeat(448,axis=0)
        base_y = np.expand_dims(a,1).repeat(448,axis=1)
        
        base_coordinate = np.concatenate((base_y.reshape(448,448,1),base_x.reshape(448,448,1)),axis=-1)
        sm = np.zeros_like(bm)
        for i in range(bm.shape[0]):
            for j in range(bm.shape[1]):
                sm[i,j] = np.array(bm[i,j][0]-i,bm[i,j][1]-w)


        # prepare content mask
        pdf = cv2.imread(pdf_path)
        pdf = cv2.cvtColor(pdf,cv2.COLOR_BGR2RGB)
        bm_tmp = bm.astype(np.float32)
        bm0_tmp = bm_tmp[:,:,0]
        bm1_tmp = bm_tmp[:,:,1]
        uw = cv2.remap(pdf.astype(np.uint8),bm0_tmp,bm1_tmp,cv2.INTER_CUBIC)
        _,content_mask = cv2.threshold(cv2.cvtColor(uw,cv2.COLOR_RGB2GRAY),240,255,cv2.THRESH_BINARY_INV)

        # prepare im
        im = m.imread(im_path,mode='RGB')
        im = np.array(im, dtype=np.uint8)

        # prepare depth
        wc_path = pjoin(self.root, 'wc', im_name + '.exr')
        wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = cv2.split(wc)[0]
        depth = np.array(depth, dtype=np.float)

        # prepare mask and edge
        mask = ((wc[:,:,0]!=0)&(wc[:,:,1]!=0)&(wc[:,:,2]!=0)).astype(np.uint8)*255
        mask_blur = cv2.blur(mask,(3,3))
        edge = cv2.Canny(mask_blur,20,150)


        # for showing the datas:

        if False:       
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(depth, mask)
            depth_show = (depth-minVal)/(maxVal-minVal)*(mask/255)
            im_show = im.transpose((2,0,1))
            im_show = torch.from_numpy(im_show)
            im_show = im_show.unsqueeze(0)
            pdf_show = pdf.transpose((2,0,1))
            pdf_show = torch.from_numpy(pdf_show)
            pdf_show = pdf_show.unsqueeze(0)
            # bm_show = bm.transpose((2,0,1))

            bm_cv = (bm/448.-0.5)*2
            bm0_cv = bm_cv[:,:,0]
            bm1_cv = bm_cv[:,:,1]
            debug = cv2.remap(pdf,bm0_cv,bm1_cv)

            bm_show = torch.from_numpy(bm)
            bm_show = bm_show.unsqueeze(0)
            bm_show = (bm_show/448.-0.5)*2
            uw_show = F.grid_sample(input=pdf_show.float(),grid=bm_show.float())
            uw_show = uw_show[0].numpy().transpose((1,2,0)).astype(np.uint8)
            
            content_mask1 = cv2.adaptiveThreshold(cv2.cvtColor(uw_show,cv2.COLOR_RGB2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,3)
            _,content_mask2 = cv2.threshold(cv2.cvtColor(uw_show,cv2.COLOR_RGB2GRAY),0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
            _,content_mask3 = cv2.threshold(cv2.cvtColor(uw_show,cv2.COLOR_RGB2GRAY),240,255,cv2.THRESH_BINARY_INV)                       
            cv2.imshow('uw',uw_show)
            cv2.imshow('try',debug)
            cv2.imshow('input', im)
            # cv2.imshow('content1', content_mask1)
            # cv2.imshow('content2', content_mask2)
            # cv2.imshow('content3', content_mask3)
            cv2.imshow('pdf', pdf)
            # cv2.imshow('edge',edge)
            # cv2.imshow('mask',mask)
            # cv2.imshow('depth',depth_show)
            cv2.waitKey()
            cv2.destroyAllWindows()




        

#        if 'val' in self.split:
#            im, depth=tight_crop(im/255.0,depth)
#            im, wc=tight_crop(im/255.0,wc)
#        if self.augmentations:          #this is for training, default false for validation\
#            tex_id=random.randint(0,len(self.txpths)-1)
#            txpth=self.txpths[tex_id] 
#            tex=cv2.imread(os.path.join(self.root[:-7],txpth)).astype(np.uint8)
#            bg=cv2.resize(tex,self.img_size,interpolation=cv2.INTER_NEAREST)
#            im,depth,msk=data_aug(im,depth,bg)
        if self.is_transform:
            im_org, im, depth_org,depth, mask, edge, bm_org, bm,content_mask_org = self.transform(im, depth, mask, edge, bm,content_mask)
        return im_org, im, depth_org, depth, mask, edge, bm_org,bm,content_mask_org


    def transform(self, img, depth, mask, edge,bm,content_mask):
        # img 
        # if img.shape[-1] == 4:
            # img=img[:,:,:3]   # Discard the alpha channel  
        # img = img[:, :, ::-1] # RGB -> BGR
        # img = img.transpose(2, 0, 1) # NHWC -> NCHW

        #depth
        xmx, xmn, ymx, ymn,zmx, zmn= 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497   # calculate from all the wcs
        depth = (depth-zmn)/(zmx-zmn)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(depth, mask)
        depth = (depth-minVal)/(maxVal-minVal)*(mask/255)
        # depth_org = depth
        # depth = cv2.resize(depth, self.img_size)

        # mask
        mask = mask.astype(float)/255.

        # edge
        edge = edge.astype(float)/255.
        edge[edge > 0.1] = 1.
        edge[edge <= 0.1] = 0.

        bm = bm.astype(float)
        # print(bm.max(),bm.min())
        bm=bm/448.0

        #content_mask
        content_mask = content_mask.astype(float)/255
        content_mask[content_mask > 0.1] =1.
        content_mask[content_mask <= 0.1 ] =0.



        # to torch
        img = img.astype(float) / 255.0
        img = img[:, :, ::-1]
        img_org = img.copy()
        img = cv2.resize(img, self.img_size)
        img_org = img_org.transpose((2,0,1))
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img).float()
        img_org = torch.from_numpy(img_org).float()
        # print(img.shape,'img')
        # print(self.img_size,type(self.img_size))

        depth_org = depth.copy()
        depth = cv2.resize(depth,self.img_size)
        depth = torch.from_numpy(depth).float()
        depth_org = torch.from_numpy(depth_org).float()
        depth = torch.unsqueeze(depth,0)
        depth_org = torch.unsqueeze(depth_org,0)
        # print(depth.shape,'depth')

        mask = cv2.resize(mask,self.img_size)
        mask = torch.from_numpy(mask).float()
        mask = torch.unsqueeze(mask,0)
        # print(mask.shape,'mask')
        # mask = F.interpolate(mask, self.img_size)

        edge = cv2.resize(edge,self.img_size)
        edge = torch.from_numpy(edge).float()
        edge = torch.unsqueeze(edge,0)
        # print(edge.shape,'edge')
        # edge = F.interpolate(edge, self.img_size)

        content_mask = torch.from_numpy(content_mask).float()
        content_mask = torch.unsqueeze(content_mask,0)

        # bm = bm.transpose((1,2,0))
        bm_org = bm.copy()
        bm = cv2.resize(bm,self.img_size)
        bm_org = cv2.resize(bm_org,self.img_size)
        bm = bm.transpose((2,0,1))
        bm = torch.from_numpy(bm).float()
        bm_org = bm_org.transpose((2,0,1))
        bm_org = torch.from_numpy(bm_org).float()
        # print(bm.shape,'bm')
        # print(img_org.shape,img.shape,depth_org.shape,depth.shape,mask.shape,edge.shape,bm_org.shape,bm.shape)
        return img_org,img,depth_org,depth,mask,edge,bm_org,bm,content_mask

