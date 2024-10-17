import numpy as np 
import cv2 
import os 
import glob
import torch
import torch.nn.functional as F
import cv2 
from tqdm import tqdm
import random
import argparse

def torch2cvimg(tensor,min=0,max=1):
    '''
    input:
        tensor -> torch.tensor BxCxHxW C can be 1,3
    return
        im -> ndarray uint8 HxWxC 
    '''
    im_list = []
    for i in range(tensor.shape[0]):
        im = tensor.detach().cpu().data.numpy()[i]
        im = im.transpose(1,2,0)
        im = np.clip(im,min,max)
        im = ((im-min)/(max-min)*255).astype(np.uint8)
        im_list.append(im)
    return im_list
def cvimg2torch(img,min=0,max=1):
    '''
    input:
        im -> ndarray uint8 HxWxC 
    return
        tensor -> torch.tensor BxCxHxW 
    '''
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1) # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    return img

def grid_norm(grid):
    grid0 = grid[:,:,0]
    grid1 = grid[:,:,1]
    grid0 = (grid0-grid0.min())/(grid0.max()-grid0.min())
    grid0 = (grid0-0.5)*2
    grid1 = (grid1-grid1.min())/(grid1.max()-grid1.min())
    grid1 = (grid1-0.5)*2
    grid = np.stack((grid0,grid1),axis=-1)
    grid = np.clip(grid,-1,1)   
    return grid 

def grid_normv2(grid):
    h,w = grid.shape[:2]

    grid0 = grid[:,:,0]
    grid1 = grid[:,:,1]

    # grid0 = (grid0-grid0.min())/2
    # import pdb;pdb.set_trace()
    # factor0 = -(1-grid0.max())/(grid0.max()-0.5)
    # grid0 = grid0 - (grid0-0.5)*factor0
    
    # grid1 = (grid1-grid1.min())/2
    # factor1 = -(1-grid1.max())/(grid1.max()-0.5)
    # grid1 = grid1 - (grid1-0.5)*factor1

    # grid0 = (grid0-0.5)*2
    # grid1 = (grid1-0.5)*2
    grid = np.stack((grid0,grid1),axis=-1)
    grid = np.clip(grid,-1,1)   
    return grid 


def grid_sum(grid1,grid2):
    '''
    input: 
        grid1 -> ndarray HxWx2 -1~1
        grid2 -> ndarray HxWx2 -1~1
    return:
        grid3 -> ndarray HxWx2 -1~1
    '''
    h,w = grid1.shape[:2]

    grid1 = (grid1+1)/2 * np.array([w,h]).reshape(1,1,2)
    grid2 = grid_normv2(grid2)
    grid2 = ((grid2+1)/2 * np.array([w,h]).reshape(1,1,2)).astype(np.int16)
    grid2 = np.clip(grid2,0,h-1)
    grid3 = np.zeros_like(grid2)

    # for i in range(h):
    #     for j in range(w):
    #         grid3[j,i] = grid1[grid2[j,i][1],grid2[j,i][0]]
    grid3 = grid1[grid2[:,:,1],grid2[:,:,0]]

    grid3 = grid3 / np.array([w,h]).reshape(1,1,2)
    grid3 = (grid3-0.5)*2


    # grid4_0, grid4_1 = np.meshgrid(np.arange(w), np.arange(h))
    # grid4 = np.stack((grid4_0/w,grid4_1/h),axis=-1)
    # grid4 = (grid4-0.5)*2*0.9090909
    # grid4 = ((grid4+1)/2 * np.array([w,h]).reshape(1,1,2)).astype(np.int16)

    # grid5 = grid3[grid4[:,:,1],grid4[:,:,0]]

    return grid3

def grid_sumv2(grid1,grid2):
    '''
    input: 
        grid1 -> ndarray HxWx2 -1~1
        grid2 -> ndarray HxWx2 -1~1
    return:
        grid3 -> ndarray HxWx2 -1~1
    '''
    h,w = grid1.shape[:2]

    grid1 = torch.from_numpy(grid1).permute(2,0,1).unsqueeze(0 )
    grid2 = torch.from_numpy(grid2).unsqueeze(0)
    grid3 = F.grid_sample(grid1,grid2)
    grid3 = grid3[0].permute(1,2,0).data.numpy()


    return grid3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--im_folder', nargs='?', type=str, default='./data/dataset1/all_data/')
    parser.set_defaults()
    args = parser.parse_args()
    grid1_paths = glob.glob(os.path.join(args.im_folder,'*_grid1*'))
    random.shuffle(grid1_paths)
    for grid1_path in tqdm(grid1_paths):
        grid2_path = grid1_path.replace('_grid1','_grid2')
        im_org_path = grid1_path.replace('_grid1','_origin').replace('.npy','')
        im_target_path = grid1_path.replace('_grid1','_target').replace('.npy','')
        im_org = cv2.imread(im_org_path)
        size = 1024

        grid1 = np.load(grid1_path)
        grid2 = np.load(grid2_path)

        grid1_0 = cv2.resize(grid1[:,:,0],(size,size))
        grid1_1 = cv2.resize(grid1[:,:,1],(size,size))
        grid1 = np.stack((grid1_0,grid1_1),axis=-1)
        grid2_0 = cv2.resize(grid2[:,:,0],(size,size))
        grid2_1 = cv2.resize(grid2[:,:,1],(size,size))
        grid2 = np.stack((grid2_0,grid2_1),axis=-1)

        grid3 = grid_sumv2(grid1,grid2)
        grid3 = grid_normv2(grid3)
    
        grid3_temp = torch.from_numpy(grid3).float().unsqueeze(0)
        grid3_temp = F.interpolate(grid3_temp.permute(0,3,1,2),im_org.shape[:2],mode='bilinear')
        dewarp = torch2cvimg(F.grid_sample(cvimg2torch(im_org),grid3_temp.permute(0,2,3,1)))[0]
        # cv2.imwrite(grid1_path.replace('_grid1.jpg.npy','_out_sum.jpg'),dewarp)

        grid3_0 = cv2.resize(grid3[:,:,0],(1024,1024))
        grid3_1 = cv2.resize(grid3[:,:,1],(1024,1024))
        grid3 = np.stack((grid3_0,grid3_1),axis=-1)
        np.save(grid1_path.replace('_grid1','_grid3'),grid3.astype(np.float16))



        # cv2.imshow('dewarp_sum',cv2.resize(dewarp,(512,512)))
        # cv2.imshow('dewarp_seperate',cv2.resize(im_dewarp,(512,512)))
        # cv2.imshow('capture',cv2.resize(im_capture,(512,512)))
        # cv2.imshow('origin',cv2.resize(im_org,(512,512)))
        # cv2.waitKey(0)
