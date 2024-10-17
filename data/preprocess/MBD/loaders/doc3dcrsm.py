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

import scipy.spatial.qhull as qhull

class doc3dcrLoader(data.Dataset):
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


        # prepare im
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im_resize = cv2.resize(im,self.img_size)
        pdf = cv2.imread(pdf_path)
        pdf = cv2.cvtColor(pdf,cv2.COLOR_BGR2RGB)
        pdf_resize = cv2.resize(pdf,self.img_size)
        
        # prepare bm
        bm = np.load(bm_path) # -> ndarray (448*448*2) 0~448 float
        bm_tmp = bm.copy()
        bm = bm/448.
        bm_resize = cv2.resize(bm, self.img_size)

        # # prepare offset map
        # uw_im1 = np.zeros_like(im)
        # base = np.arange(448)
        # base_x = np.expand_dims(base,0).repeat(448,axis=0)
        # base_y = np.expand_dims(base,1).repeat(448,axis=1)
        # base_coordinate = np.concatenate((base_y.reshape(448,448,1),base_x.reshape(448,448,1)),axis=-1)
        # # base_coordinate -> [0,0],[0,1],[0,2]
        # om = np.zeros_like(bm_tmp)
        # for i in range(bm_tmp.shape[0]):
        #     for j in range(bm_tmp.shape[1]):
        #         # om[int(bm_tmp[i,j][1]),int(bm_tmp[i,j][0])] = np.array([j-bm_tmp[i,j][1],i-bm_tmp[i,j][0]])
        #         om[int(bm_tmp[i,j][1]),int(bm_tmp[i,j][0])] = np.array([i-bm_tmp[i,j][1],j-bm_tmp[i,j][0]])
        # def interp_weights(xyz, uvw):
        #     # xyz: pixel_position
        #     # uvw: original_pixel_position
        #     tri = qhull.Delaunay(xyz)
        #     simplex = tri.find_simplex(uvw)
        #     vertices = np.take(tri.simplices, simplex, axis=0)
        #     # pixel_triangle = pixel[tri.simplices]
        #     temp = np.take(tri.transform, simplex, axis=0)
        #     delta = uvw - temp[:, 2]
        #     bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
        #     return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
        # def interpolate(values, vtx, wts):
        #     return np.einsum('njk,nj->nk', np.take(values, vtx, axis=0), wts)
        # origin_pixel_position = base_coordinate.reshape(448*448,2)
        # perturbed_label = om.reshape(448*448,2)
        # perturbed_label_classify = np.ones((448*448,2))*255
        # # print(perturbed_label)
        # perturbed_label_classify[perturbed_label==[0.,0.]] = 0
        # perturbed_label_classify = perturbed_label_classify[:,0]
        # # print(perturbed_label.min(),perturbed_label.max(),perturbed_label[0:5],'==============')
        # perturbed_img = im.reshape(448*448, 3)
        # perturbed_label += origin_pixel_position
        # # perturbed_label += origin_pixel_position
        # pixel_position = perturbed_label[perturbed_label_classify != 0, :]
        # pixel = perturbed_img[perturbed_label_classify != 0, :]
        # '''construct Delaunay triangulations in all scattered pixels and then using interpolation'''
        # vtx, wts = interp_weights(pixel_position, origin_pixel_position)
        # # wts[np.abs(wts)>=1]=0
        # wts_sum = np.abs(wts).sum(-1)
        # wts = wts[wts_sum <= 1, :]
        # vtx = vtx[wts_sum <= 1, :]
        # flat_img = np.zeros_like(perturbed_img)
        # # flat_img.reshape(448 * 448, 3)[wts_sum <= 1, :] = interpolate(pixel, vtx, wts)
        # flat_img.reshape(448 * 448, 3)[wts_sum <= 1, :] = interpolate(pixel, vtx, wts)
        # flat_img = flat_img.reshape(448, 448, 3)

        # prepare crop offset map
        im_crop = (im[149:298,149:298]).copy()
        # im_crop = (im[112:298,149:298]).copy()
        # im_crop = cv2.resize(im_crop,(448,448))
        base = np.arange(448)
        base_x = np.expand_dims(base,0).repeat(448,axis=0)
        base_y = np.expand_dims(base,1).repeat(448,axis=1)
        base_coordinate = np.concatenate((base_y.reshape(448,448,1),base_x.reshape(448,448,1)),axis=-1)
        base_coordinate_crop = base_coordinate
        # base_coordinate_crop = (base_coordinate[149:298,149:298]).copy()
        # base_coordinate -> [0,0],[0,1],[0,2]
        bm_tmp[bm_tmp[:,:,:]>298]=0
        bm_tmp[bm_tmp[:,:,:]<149]=0
        om = np.zeros_like(bm_tmp)
        for i in range(bm_tmp.shape[0]):
            for j in range(bm_tmp.shape[1]):
                om[int(bm_tmp[i,j][1]),int(bm_tmp[i,j][0])] = np.array([i-bm_tmp[i,j][1],j-bm_tmp[i,j][0]])
                # om[int(bm_tmp[i,j][1]),int(bm_tmp[i,j][0])] = np.array([i-bm_tmp[i,j][1]-50,j-bm_tmp[i,j][0]-50])
                # om[int(bm_tmp[i,j][1]),int(bm_tmp[i,j][0])] = np.array([i-bm_tmp[i,j][1]-np.sign(i-bm_tmp[i,j][1])*149,j-bm_tmp[i,j][0]])
                # if 149<i<298 and 149<j<298:
                #     print([i,j],i-bm_tmp[i,j][1],np.sign(i-bm_tmp[i,j][1])*149,i-bm_tmp[i,j][1]-np.sign(i-bm_tmp[i,j][1])*149)

        # om_crop = (om[149:298,149:298]).copy()/5
        # normalize_max = max(abs(om_crop.min()),om_crop.max())
        
        
        # om_crop = cv2.resize(om_crop,(448,448))
        def interp_weights(xyz, uvw):
            # xyz: pixel_position
            # uvw: original_pixel_position
            tri = qhull.Delaunay(xyz)
            simplex = tri.find_simplex(uvw)
            vertices = np.take(tri.simplices, simplex, axis=0)
            # pixel_triangle = pixel[tri.simplices]
            temp = np.take(tri.transform, simplex, axis=0)
            delta = uvw - temp[:, 2]
            bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
            return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
        def interpolate(values, vtx, wts):
            return np.einsum('njk,nj->nk', np.take(values, vtx, axis=0), wts)
        origin_pixel_position = base_coordinate.reshape(448*448,2)
        perturbed_label = (om.reshape(448*448,2)).copy()
        perturbed_label_classify = np.ones((448*448,2))*255
        # print(perturbed_label)
        perturbed_label_classify[perturbed_label==[0.,0.]] = 0
        perturbed_label_classify = perturbed_label_classify[:,0]
        # print(perturbed_label.min(),perturbed_label.max(),perturbed_label[0:5],'==============')
        perturbed_img = im.reshape(448*448, 3)
        # perturbed_label = origin_pixel_position
        perturbed_label += origin_pixel_position
        # perturbed_label = perturbed_label/3
        pixel_position = perturbed_label[perturbed_label_classify != 0, :]
        pixel = perturbed_img[perturbed_label_classify != 0, :]

        '''construct Delaunay triangulations in all scattered pixels and then using interpolation'''
        vtx, wts = interp_weights(pixel_position, origin_pixel_position)
        # wts[np.abs(wts)>=1]=0
        wts_sum = np.abs(wts).sum(-1)
        wts = wts[wts_sum <= 1, :]
        vtx = vtx[wts_sum <= 1, :]
        flat_img = np.zeros_like(perturbed_img)
        # flat_img.reshape(149 * 149, 3)[wts_sum <= 1, :] = interpolate(pixel, vtx, wts)
        flat_img.reshape(448 * 448, 3)[wts_sum <= 1, :] = interpolate(pixel, vtx, wts)
        flat_img = flat_img.reshape(448, 448, 3)
        # flat_img_crop = flat_img[149:298,149:298]


        cv2.imshow('im',im)
        cv2.imshow('im_crop',im_crop)
        cv2.imshow('flat',flat_img.astype(np.uint8))
        # cv2.imshow('flat_crop',flat_img_crop.astype(np.uint8))
        # cv2.imshow('y',((om[:,:,0]-om[:,:,0].min())/(om[:,:,0].max()-om[:,:,0].min())*255).astype(np.uint8))
        # cv2.imshow('y1',((bm_tmp[:,:,0]-bm_tmp[:,:,0].min())/(bm_tmp[:,:,0].max()-bm_tmp[:,:,0].min())*255).astype(np.uint8))
        # cv2.imshow('x',((om[:,:,1]-om[:,:,1].min())/(om[:,:,1].max()-om[:,:,1].min())*255).astype(np.uint8))
        # cv2.imshow('x1',((bm_tmp[:,:,1]-bm_tmp[:,:,1].min())/(bm_tmp[:,:,1].max()-bm_tmp[:,:,1].min())*255).astype(np.uint8))
        # cv2.imshow('x',uw_im_baseom)
        cv2.waitKey()
        cv2.destroyAllWindows()



        # prepare unwarp image
        bm0_tmp = bm_tmp[:,:,0].astype(np.float32)
        bm1_tmp = bm_tmp[:,:,1].astype(np.float32)
        uw_pdf = cv2.remap(pdf.astype(np.uint8),bm0_tmp,bm1_tmp,cv2.INTER_CUBIC)
        uw_im = cv2.remap(im.astype(np.uint8),bm0_tmp,bm1_tmp,cv2.INTER_CUBIC)
        uw_im_resize = cv2.resize(uw_im,self.img_size)
        uw_pdf_resize = cv2.resize(uw_pdf,self.img_size)


        # prepare content mask
        _,content_mask = cv2.threshold(cv2.cvtColor(uw_pdf,cv2.COLOR_RGB2GRAY),240,255,cv2.THRESH_BINARY_INV)
        content_mask_resize = cv2.resize(content_mask,self.img_size)

        # cv2.imshow('bm',(bm[:,:,0]*255).astype(np.uint8))
        # cv2.imshow('bm_resize',(bm_resize[:,:,0]*255).astype(np.uint8))
        # cv2.imshow('pdf',uw_pdf)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # prepare depth
        # wc_path = pjoin(self.root, 'wc', im_name + '.exr')
        # wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        # depth = cv2.split(wc)[0]
        # depth = np.array(depth, dtype=np.float)
        # prepare mask and edge
        # mask = ((wc[:,:,0]!=0)&(wc[:,:,1]!=0)&(wc[:,:,2]!=0)).astype(np.uint8)*255
        # mask_blur = cv2.blur(mask,(3,3))
        # edge = cv2.Canny(mask_blur,20,150)

        

#        if 'val' in self.split:
#            im, depth=tight_crop(im/255.0,depth)
#            im, wc=tight_crop(im/255.0,wc)
#        if self.augmentations:          #this is for training, default false for validation\
#            tex_id=random.randint(0,len(self.txpths)-1)
#            txpth=self.txpths[tex_id] 
#            tex=cv2.imread(os.path.join(self.root[:-7],txpth)).astype(np.uint8)
#            bg=cv2.resize(tex,self.img_size,interpolation=cv2.INTER_NEAREST)
#            im,depth,msk=data_aug(im,depth,bg)
        im = self.transform(im)
        uw_im = self.transform(uw_im)
        uw_pdf = self.transform(uw_pdf)
        content_mask = self.transform(content_mask)
        
        # if self.is_transform:
        #     im_org, im, depth_org,depth, mask, edge, bm_org, bm,content_mask_org = self.transform(im, depth, mask, edge, bm,content_mask)
        return im,bm,content_mask,uw_im

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

