from os.path import join as pjoin
import collections
import torch
import numpy as np
import cv2
import random
from torch.utils import data
import glob

class docalign12kLoader(data.Dataset):
    def __init__(self, root, split='train', is_transform=False,
                 img_size=512, augmentations=None):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 3   
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.size = self.img_size[0]
        path = pjoin(self.root, split + '.txt')
        file_list = tuple(open(path, 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[split] = file_list
        self.shadow_paths = glob.glob('./data/DocAlign12K/shadows/*')

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]   # 1/824_8-cp_Page_0503-7Nw0001
        flat_path = pjoin(self.root, 'flat',  im_name+ '.jpg') 
        warp_path = pjoin(self.root, 'distorted_hard',  im_name+ '.jpg') 
        map_path = pjoin(self.root, 'forwardmap_hard',  im_name+ '.npy')
        shadow_path = random.choice(self.shadow_paths)

        ## prepare im
        flat_im = cv2.imread(flat_path)
        flat_im = cv2.resize(flat_im,self.img_size)
        warp_im = cv2.imread(warp_path)
        warp_im = cv2.resize(warp_im,self.img_size)
        shadow_im = cv2.imread(shadow_path)
        warp_im,flat_im = self.randomAugment(warp_im,flat_im,shadow_im)

        ### prepare map
        map = np.load(map_path).astype(np.float)
        # map = np.clip(map,0,1024)
        temp0 = map[:,:,0].copy()
        temp1 = map[:,:,1].copy()
        temp0 = cv2.resize(temp0,self.img_size)
        temp1 = cv2.resize(temp1,self.img_size)
        map = np.stack((temp1,temp0),axis=-1)

        # visulize
        # x, y = np.meshgrid(np.arange(1024), np.arange(1024))
        # temp0 = map[:,:,0].copy()
        # temp1 = map[:,:,1].copy()
        # temp0 = cv2.resize(temp0,(1024,1024))
        # temp1 = cv2.resize(temp1,(1024,1024))
        # map_temp = np.stack((temp0,temp1),axis=-1)
        # map_temp = map_temp.astype(np.float32)
        # x = x.astype(np.float32)
        # y = y.astype(np.float32)
        # dewarp_im = cv2.remap(warp_im,map_temp[:,:,0],map_temp[:,:,1],cv2.INTER_LINEAR)
        # cv2.imwrite('show/flat.jpg',cv2.resize(flat_im,(512,512)))
        # cv2.imwrite('show/warp.jpg',cv2.resize(warp_im,(512,512)))
        # cv2.imwrite('show/dewarp.jpg',cv2.resize(dewarp_im,(512,512)))
        # cv2.imwrite('show/high_frequency.jpg',cv2.resize(high_frequency,(512,512)))
        # exit()
        # cv2.imshow('content_mask',cv2.resize(content_mask,(512,512)))
        # cv2.waitKey(0)

        warp_im = self.transform(warp_im)
        flat_im = self.transform(flat_im)

        x, y = np.meshgrid(np.arange(1024),np.arange(1024))
        base_cord = np.stack((x,y),axis=-1)
        flow = map-base_cord

        flow = flow.transpose(2, 0, 1)
        flow = torch.from_numpy(flow)        
        return warp_im, flat_im, flow

    def transform(self,img):
        if len(img.shape) == 2 :
            img = np.expand_dims(img,-1)
        img = img.astype(np.float)/255. 
        img = (img-0.5)*2
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        return img

    def randomAugment(self,in_img,gt_img,shadow_img):
        h,w = in_img.shape[:2]
        # random crop
        crop_size = random.randint(128,1024)
        if shadow_img.shape[0] <= crop_size:
            shadow_img = cv2.copyMakeBorder(shadow_img,crop_size-shadow_img.shape[0]+1,0,0,0,borderType=cv2.BORDER_CONSTANT,value=(128,128,128))
        if shadow_img.shape[1] <= crop_size:
            shadow_img = cv2.copyMakeBorder(shadow_img,0,0,crop_size-shadow_img.shape[1]+1,0,borderType=cv2.BORDER_CONSTANT,value=(128,128,128))

        shift_y = np.random.randint(0,shadow_img.shape[1]-crop_size)
        shift_x = np.random.randint(0,shadow_img.shape[0]-crop_size)
        shadow_img = shadow_img[shift_x:shift_x+crop_size,shift_y:shift_y+crop_size,:]
        shadow_img = cv2.resize(shadow_img,(w,h))
        in_img = in_img.astype(np.float64)*(shadow_img.astype(np.float64)+1)/255
        in_img = np.clip(in_img,0,255).astype(np.uint8)

        ## brightness
        if random.uniform(0,1) <= 0.5:
            high = 1.3
            low = 0.8
            ratio = np.random.uniform(low,high)
            in_img = in_img.astype(np.float64)*ratio
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## contrast
        if random.uniform(0,1) <= 0.5:
            high = 1.3
            low = 0.8
            ratio = np.random.uniform(low,high)
            gray = cv2.cvtColor(in_img,cv2.COLOR_BGR2GRAY)
            mean = np.mean(gray)
            mean_array = np.ones_like(in_img).astype(np.float64)*mean
            in_img = in_img.astype(np.float64)*ratio + mean_array*(1-ratio)
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## color
        if random.uniform(0,1) <= 0.5:
            high = 0.2
            low = 0.1
            ratio = np.random.uniform(0.1,0.3)
            random_color = np.random.randint(50,200,3).reshape(1,1,3)
            random_color = (random_color*ratio).astype(np.uint8)
            random_color = np.tile(random_color,(self.img_size[0],self.img_size[1],1))
            in_img = in_img.astype(np.float64)*(1-ratio) + random_color
            in_img = np.clip(in_img,0,255).astype(np.uint8)        
        ## scale and rotate
        if random.uniform(0,1) <= 0:
            y,x = self.img_size
            angle = random.uniform(-180,180)
            scale = random.uniform(0.5,1.5)
            M = cv2.getRotationMatrix2D((int(x/2),int(y/2)),angle,scale)
            in_img = cv2.warpAffine(in_img,M,(x,y),borderValue=0)
            gt_img = cv2.warpAffine(gt_img,M,(x,y),borderValue=0)
        # add noise
        ## jpegcompression
        # if 'test' in self.split:
        #     return in_img, gt_img
            
        quanlity_high = 95
        quanlity_low = 45
        quanlity = int(np.random.randint(quanlity_low,quanlity_high))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),quanlity]
        result, encimg = cv2.imencode('.jpg',in_img,encode_param)
        in_img = cv2.imdecode(encimg,1).astype(np.uint8)
        ## gaussiannoise
        mean = 0
        sigma = 0.02
        noise_ratio = 0.004
        num_noise = int(np.ceil(noise_ratio*w))
        coords = [np.random.randint(0,i-1,int(num_noise)) for i in [h,w]] 
        gauss = np.random.normal(mean,sigma,num_noise*3)*255
        guass = np.reshape(gauss,(-1,3))
        in_img = in_img.astype(np.float64)
        in_img[tuple(coords)] += guass
        in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## blur
        ksize = np.random.randint(1,2)*2 + 1
        in_img = cv2.blur(in_img,(ksize,ksize))
        return in_img, gt_img
