#test end to end benchmark data test
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.autograd import Variable
from tqdm import tqdm
import glob 


from models import get_model
from utils import cvimg2torch, torch2cvimg,flow2normmap,get_sobel
from loss_metric import smooth
import time
import random


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def self_ensembel(img, grid_paths, num):
    h,w = img.shape[:2]
    grid_paths = random.sample(grid_paths,num)
    grids = torch.zeros((num,2,h,w))
    for i, grid_path in enumerate(grid_paths):
        base = np.stack(np.meshgrid(np.arange(1024),np.arange(1024)),axis=-1)
        grid = np.load(grid_path).astype(np.float)
        grid0 = cv2.resize(grid[:,:,0],(1024,1024))
        grid1 = cv2.resize(grid[:,:,1],(1024,1024))
        grid = np.stack((grid1,grid0),axis=-1)
        grid = (grid-base)*2+base

        grid = grid/1024.
        grid = torch.from_numpy(grid).unsqueeze(0).permute(0,3,1,2).float()
        grid = F.interpolate(grid,(h,w),mode='bilinear')
        grid0 = grid[:,0,:,:]
        grid1 = grid[:,1,:,:]
        grid0 = (grid0-grid0.min())/(grid0.max()-grid0.min())
        grid1 = (grid1-grid1.min())/(grid1.max()-grid1.min())
        grid0 = (grid0-0.5)*2
        grid1 = (grid1-0.5)*2
        grids[i,0,:,:] = grid0
        grids[i,1,:,:] = grid1
    imgs = torch.tile(cvimg2torch(img),(num,1,1,1))
    imgs = F.grid_sample(imgs,grids.permute(0,2,3,1))
    imgs = torch2cvimg(imgs)
    imgs.append(img)
    return imgs

def self_augment(img, grid_paths):
    h,w = img.shape[:2]
    grid_path = random.choice(grid_paths)

    grids = torch.zeros((1,2,h,w))
    base = np.stack(np.meshgrid(np.arange(1024),np.arange(1024)),axis=-1)
    grid = np.load(grid_path).astype(np.float)
    grid0 = cv2.resize(grid[:,:,0],(1024,1024))
    grid1 = cv2.resize(grid[:,:,1],(1024,1024))
    grid = np.stack((grid1,grid0),axis=-1)
    weight = random.choice([0,1,2])
    grid = (grid-base)*weight+base
    grid = grid/1024.
    grid = torch.from_numpy(grid).unsqueeze(0).permute(0,3,1,2).float()
    grid = F.interpolate(grid,(h,w),mode='bilinear')
    grid0 = grid[:,0,:,:]
    grid1 = grid[:,1,:,:]
    grid0 = (grid0-grid0.min())/(grid0.max()-grid0.min())
    grid1 = (grid1-grid1.min())/(grid1.max()-grid1.min())
    grid0 = (grid0-0.5)*2
    grid1 = (grid1-0.5)*2
    grids[0,0,:,:] = grid0
    grids[0,1,:,:] = grid1
    imgs = torch.tile(cvimg2torch(img),(1,1,1,1))
    imgs = F.grid_sample(imgs,grids.permute(0,2,3,1))
    imgs = torch2cvimg(imgs)
    return imgs[0]



def unwarp(img, bm,h_org,w_org):
    w,h=img.shape[0],img.shape[1]
    bm = bm.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()[0,:,:,:]
    bm0=cv2.blur(bm[:,:,0],(3,3))
    bm1=cv2.blur(bm[:,:,1],(3,3))
    bm0=cv2.resize(bm0,(w_org,h_org))
    bm1=cv2.resize(bm1,(w_org,h_org))
    bm=np.stack([bm0,bm1],axis=-1)
    bm=np.expand_dims(bm,0)
    bm=torch.from_numpy(bm).float().cuda()

    img = img.astype(float) / 255.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float().cuda()

    res = F.grid_sample(input=img, grid=bm,padding_mode="border")
    res = (res[0].cpu().data.numpy().transpose(1,2,0)*255).astype(np.uint8)
    return res
def remap_using_flow_fields(image, disp_x, disp_y, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT):
    """
    opencv remap : carefull here mapx and mapy contains the index of the future position for each pixel
    not the displacement !
    map_x contains the index of the future horizontal position of each pixel [i,j] while map_y contains the index of the future y
    position of each pixel [i,j]

    All are numpy arrays
    :param image: image to remap, HxWxC
    :param disp_x: displacement on the horizontal direction to apply to each pixel. must be float32. HxW
    :param disp_y: isplacement in the vertical direction to apply to each pixel. must be float32. HxW
    :return:
    remapped image. HxWxC
    """
    h_scale, w_scale=image.shape[:2]
    disp_x = cv2.resize(disp_x,(w_scale,h_scale),interpolation=cv2.INTER_LINEAR)/1024*w_scale
    disp_y = cv2.resize(disp_y,(w_scale,h_scale),interpolation=cv2.INTER_LINEAR)/1024*h_scale
    # estimate the grid
    X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                       np.linspace(0, h_scale - 1, h_scale))
    map_x = (X+disp_x).astype(np.float32)
    map_y = (Y+disp_y).astype(np.float32)
    remapped_image = cv2.remap(image, map_x, map_y, interpolation=interpolation, borderMode=border_mode)
    grid = np.stack((map_x/w_scale,map_y/h_scale),axis=-1)
    grid = (grid-0.5)*2

    # grid0 = grid[:,:,0]
    # grid0 = cv2.resize(grid0,(128,128))
    # grid0 = cv2.blur(grid0,(9,9))
    # grid0 = cv2.resize(grid0,(w_scale,h_scale),interpolation=cv2.INTER_LINEAR)
    # grid1 = grid[:,:,1]
    # grid1 = cv2.resize(grid1,(128,128))
    # grid1 = cv2.blur(grid1,(9,9))
    # grid1 = cv2.resize(grid1,(w_scale,h_scale),interpolation=cv2.INTER_LINEAR)
    # grid = np.stack((grid0,grid1),axis=-1)

    return remapped_image,grid
def map_norm(grid):
    '''
    grid -> bx2xhxw 
    '''
    all_min = torch.min(torch.min(grid,dim=2,keepdim=True)[0],dim=3,keepdim=True)[0].detach()
    all_max = torch.max(torch.max(grid,dim=2,keepdim=True)[0],dim=3,keepdim=True)[0].detach()
    
    new_grid = (grid - all_min)/(all_max-all_min)
    # for batch in range(grid.shape[0]):
    #     grid[batch][0] = ((grid[batch][0]-grid[batch][0].min())/(grid[batch][0].max()-grid[batch][0].min())-0.5)*2
    #     grid[batch][1] = ((grid[batch][1]-grid[batch][1].min())/(grid[batch][1].max()-grid[batch][1].min())-0.5)*2
    return new_grid


def optimize(args,warp_path):

    BATCHSIZE=args.batchsize
    # Predict
    # model = get_model('glu', n_classes=2, in_channels=6, img_size=args.img_rows)
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # model.cuda()
    # checkpoint = torch.load(args.model_path,map_location='cpu')
    ttt = time.time()
    if not args.model_path is None:
        model.load_state_dict(checkpoint['model_state'])
    print(time.time()-ttt)
    # model.load_state_dict(checkpoint['state_dict'])
    optimizer= torch.optim.Adam(model.parameters(),lr=1e-4)
    # sched = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5,step_size=10)


    model.train()
    min_metric = 999
    plain = 0
    for iteration in range(50):
        img_size=(1024,1024)
        flat_path = warp_path.replace('_capture.','_target.')
        flat_img_org = cv2.imread(flat_path)
        h,w = flat_img_org.shape[:2]
        h_org,w_org = flat_img_org.shape[:2]
        flat_img_resize = cv2.resize(flat_img_org, img_size)

        warp_img_org = cv2.imread(warp_path)
        warp_img_mask = warp_img_org.copy()
        # warp_img_mask[mask!=255]=0
        warp_img_mask = cv2.resize(warp_img_mask, (w_org,h_org))
        warp_img_resize = cv2.resize(warp_img_mask, img_size)
        warp_img_org = cv2.resize(warp_img_org, (w_org,h_org))

        warp_img_list = self_ensembel(warp_img_resize,glob.glob('data/augmentation_flow/forwardmap_hard/*/*'),BATCHSIZE-1)

        warp_sobels = torch.zeros(BATCHSIZE,1,1024,1024).float().cuda()
        for i, warp_img_resize in enumerate(warp_img_list):
            # cv2.imshow('1',cv2.resize(warp_img_resize,(512,512)))
            # cv2.waitKey(0)
            warp_sobels[i] = cvimg2torch(get_sobel(warp_img_resize)).float().cuda()

        warp_imgs = torch.zeros(BATCHSIZE,3,1024,1024).float().cuda()
        for i, warp_img_resize in enumerate(warp_img_list):
            warp_img = warp_img_resize.astype(float) / 255.0
            warp_img = (warp_img-0.5)*2
            warp_img = warp_img.transpose(2, 0, 1) # NHWC -> NCHW
            warp_img = np.expand_dims(warp_img, 0)
            warp_imgs[i] = torch.from_numpy(warp_img).float().cuda()


        flat_sobel = cvimg2torch(get_sobel(flat_img_resize)).float().cuda()
        flat_img = flat_img_resize.astype(float) / 255.0
        flat_img = (flat_img-0.5)*2
        flat_img = flat_img.transpose(2, 0, 1) # NHWC -> NCHW
        flat_img = np.expand_dims(flat_img, 0)
        flat_img = torch.from_numpy(flat_img).float().cuda()
        flat_imgs = torch.tile(flat_img,(BATCHSIZE,1,1,1))
        flat_sobels = torch.tile(flat_sobel,(BATCHSIZE,1,1,1))

        # warp_sobels[-1,:,:5,:] = 1 
        # warp_sobels[-1,:,-5:,:] = 1 
        # warp_sobels[-1,:,:,-5:] = 1 
        # warp_sobels[-1,:,:,:5] = 1 
        # flat_sobels[-1,:,:5,:] = 1 
        # flat_sobels[-1,:,-5:,:] = 1 
        # flat_sobels[-1,:,:,-5:] = 1 
        # flat_sobels[-1,:,:,:5] = 1 


        l1 = nn.L1Loss()
        pred_flow4,pred_flow3,pred_flow2,pred_flow1,pred_iter = model(flat_imgs,warp_imgs,F.interpolate(flat_imgs,(256,256)),F.interpolate(warp_imgs,(256,256)))


        # loss_for234 = 0
        # for pred_flow in [pred_flow4,pred_flow3,pred_flow2]:
        #     resolution = pred_flow.shape[2]
        #     pred_map1 = flow2normmap(pred_flow/1024*resolution,size=resolution)
        #     pred_map1_0_max,pred_map1_1_max = torch.max(torch.max(pred_map1[:,0,:,:],dim=-1)[0],dim=-1)[0],torch.max(torch.max(pred_map1[:,1,:,:],dim=-1)[0],dim=-1)[0]
        #     pred_map1_0_min,pred_map1_1_min = torch.min(torch.min(pred_map1[:,0,:,:],dim=-1)[0],dim=-1)[0],torch.min(torch.min(pred_map1[:,1,:,:],dim=-1)[0],dim=-1)[0]
        #     max_gt = torch.ones_like(pred_map1_0_max)
        #     min_gt = torch.ones_like(pred_map1_0_min)*(-1)
        #     range_loss = l1(pred_map1_0_min,min_gt) + l1(pred_map1_1_min,min_gt) + l1(pred_map1_0_max,max_gt) + l1(pred_map1_1_max,max_gt)
        #     # smooth_loss = smooth.Smoothloss(pred_map1)
        #     smooth_loss = smooth.Smoothlossv2(pred_flow) 
        #     pred_sobels = F.grid_sample(F.interpolate(warp_sobels,(resolution,resolution)).float(),pred_map1.permute(0,2,3,1).float())
        #     loss_l1 = l1(pred_sobels,F.interpolate(flat_sobels,(resolution,resolution)))
        #     loss_for234 += (loss_l1*500 + range_loss*10 + + smooth_loss*0.001 + smooth.TVloss(pred_flow)*5) * (0.8**(num_iter - iter))

        
        loss = 0
        num_iter = len(pred_iter)
        for iter in range(num_iter):
            pred_map1 = flow2normmap(F.interpolate(pred_iter[iter],(1024,1024)),size=1024)
            pred_map1_0_max,pred_map1_1_max = torch.max(torch.max(pred_map1[:,0,:,:],dim=-1)[0],dim=-1)[0],torch.max(torch.max(pred_map1[:,1,:,:],dim=-1)[0],dim=-1)[0]
            pred_map1_0_min,pred_map1_1_min = torch.min(torch.min(pred_map1[:,0,:,:],dim=-1)[0],dim=-1)[0],torch.min(torch.min(pred_map1[:,1,:,:],dim=-1)[0],dim=-1)[0]
            max_gt = torch.ones_like(pred_map1_0_max)
            min_gt = torch.ones_like(pred_map1_0_min)*(-1)
            range_loss = l1(pred_map1_0_min,min_gt) + l1(pred_map1_1_min,min_gt) + l1(pred_map1_0_max,max_gt) + l1(pred_map1_1_max,max_gt)
            # smooth_loss = smooth.Smoothloss(pred_map1)
            smooth_loss = smooth.Smoothlossv2(pred_iter[iter]) 
            pred_sobels = F.grid_sample(warp_sobels.float(),pred_map1.permute(0,2,3,1).float())
            loss_l1 = l1(pred_sobels,flat_sobels)
            loss += (loss_l1*500 + range_loss*10 + smooth_loss*0.001 + smooth.TVloss(pred_iter[iter])*5) * (0.8**(num_iter - iter))
            # loss += (loss_l1*500 + range_loss*0 + smooth_loss*0.001 + smooth.TVloss(pred_iter[iter])*5) * (0.8**(num_iter - iter))
            
        loss = loss
        metric = l1(pred_sobels[-1].detach(),flat_sobels[-1].detach()).item()
        


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # sched.step()
        print('iter {}, metric {:.4f} loss {:.4f}'.format(iteration+1,metric,loss.item()))

    return model

def optimize_finetune(args,warp_paths,training_epoch=10):
    # optimizer= torch.optim.Adam(model.parameters(),lr=5e-4)
    # sched = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5,step_size=10)
    optimizer= torch.optim.Adam(model.parameters(),lr=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.2,step_size=100)        

    img_size=(1024,1024)
    BATCHSIZE=args.batchsize
    model.train()
    l1 = nn.L1Loss()
    for epoch in range(training_epoch):
        metric_dict = {'ssim':0.}
        random.shuffle(warp_paths)
        sample_num = len(warp_paths)
        iter_num = sample_num//BATCHSIZE
        for iter in range(iter_num):
            if (iter+1)*BATCHSIZE >= sample_num:
                continue 
            batch_warp_paths = warp_paths[iter*BATCHSIZE:(iter+1)*BATCHSIZE]
            
            ## get batch data
            warp_imgs = torch.zeros(BATCHSIZE,3,1024,1024).float().cuda()
            flat_imgs = torch.zeros(BATCHSIZE,3,1024,1024).float().cuda()
            warp_sobels = torch.zeros(BATCHSIZE,1,1024,1024).float().cuda()
            flat_sobels = torch.zeros(BATCHSIZE,1,1024,1024).float().cuda()
            for i, warp_path in enumerate(batch_warp_paths):
                flat_path = warp_path.replace('_capture.','_target.')

                warp_img_org = cv2.imread(warp_path)
                warp_img_resize = cv2.resize(warp_img_org.copy(), img_size)

                flat_img_org = cv2.imread(flat_path)
                flat_img_resize = cv2.resize(flat_img_org.copy(), img_size)
                warp_img_resize = self_augment(warp_img_resize,glob.glob('data/augmentation_flow/forwardmap_hard/*/*'))
                # warp_img_resize = warp_img_resize
                warp_img = warp_img_resize.astype(float) / 255.0
                warp_img = (warp_img-0.5)*2
                warp_img = warp_img.transpose(2, 0, 1) # NHWC -> NCHW
                warp_img = np.expand_dims(warp_img, 0)
                warp_imgs[i] = torch.from_numpy(warp_img).float().cuda()

                flat_img = flat_img_resize.astype(float) / 255.0
                flat_img = (flat_img-0.5)*2
                flat_img = flat_img.transpose(2, 0, 1) # NHWC -> NCHW
                flat_img = np.expand_dims(flat_img, 0)
                flat_imgs[i] = torch.from_numpy(flat_img).float().cuda()

                warp_sobels[i] = cvimg2torch(get_sobel(warp_img_resize)).float().cuda()
                flat_sobels[i] = cvimg2torch(get_sobel(flat_img_resize)).float().cuda()


            ## training 
            pred_flow4,pred_flow3,pred_flow2,pred_flow1,pred_iter = model(flat_imgs,warp_imgs,F.interpolate(flat_imgs,(256,256)),F.interpolate(warp_imgs,(256,256)))
            loss = 0
            num_iter = len(pred_iter)
            for i in range(num_iter):
                pred_map = flow2normmap(F.interpolate(pred_iter[i],(1024,1024)),size=1024)
                pred_map_0_max,pred_map_1_max = torch.max(torch.max(pred_map[:,0,:,:],dim=-1)[0],dim=-1)[0],torch.max(torch.max(pred_map[:,1,:,:],dim=-1)[0],dim=-1)[0]
                pred_map_0_min,pred_map_1_min = torch.min(torch.min(pred_map[:,0,:,:],dim=-1)[0],dim=-1)[0],torch.min(torch.min(pred_map[:,1,:,:],dim=-1)[0],dim=-1)[0]
                max_gt = torch.ones_like(pred_map_0_max)
                min_gt = torch.ones_like(pred_map_0_min)*(-1)

                range_loss1 = l1(pred_map_0_min[pred_map_0_min<min_gt],min_gt[pred_map_0_min<min_gt])
                range_loss2 = l1(pred_map_1_min[pred_map_1_min<min_gt],min_gt[pred_map_1_min<min_gt])
                range_loss3 = l1(pred_map_0_max[pred_map_0_max>max_gt],max_gt[pred_map_0_max>max_gt])
                range_loss4 = l1(pred_map_1_max[pred_map_1_max>max_gt],max_gt[pred_map_1_max>max_gt])
                for temp in [range_loss1,range_loss2,range_loss3,range_loss4]:
                    if not torch.isnan(temp):
                        loss += temp*10

                smooth_loss = smooth.Smoothlossv2(pred_iter[i]) 
                pred_sobels = F.grid_sample(warp_sobels.float(),pred_map.permute(0,2,3,1).float())

                # warp_sobel = torch2cvimg(warp_sobels.cpu())[0]
                # pred_sobel = torch2cvimg(pred_sobels.cpu())[0]
                # flat_sobel = torch2cvimg(flat_sobels.cpu())[0]
                # cv2.imshow('warp_sobel',warp_sobel)
                # cv2.imshow('pred_sobel',pred_sobel)
                # cv2.imshow('flat_sobel',flat_sobel)
                # cv2.waitKey(0)

                loss_l1 = l1(pred_sobels,flat_sobels)
                loss += (loss_l1*500 + smooth_loss*0.001 + smooth.TVloss(pred_iter[i])*5) * (0.8**(num_iter - i))

            loss = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dewarped_sobel = F.grid_sample(warp_sobels.float(),pred_iter[-1].permute(0,2,3,1).float())
            metric = l1(dewarped_sobel,flat_sobels.float()).item()
            metric_dict['ssim'] += metric

            print('epoch {}, iter {}, metric {:.4f} loss {:.4f}'.format(epoch+1, iter+1,metric,loss.item()))
        print(metric_dict['ssim']/iter_num)
        sched.step()


    return model



def test(args,warp_path,model):

    img_size=(1024,1024)

    # Setup image
    # mask_path = warp_path.replace('_capture.','_mask_new.')
    # mask = cv2.imread(mask_path)

    flat_path = warp_path.replace('_capture.','_target.')
    flat_img_org = cv2.imread(flat_path)
    # padding_w = int(w*0.05)
    # padding_h = int(h*0.05)
    # flat_img_org = cv2.copyMakeBorder(flat_img_org,padding_h,padding_h,padding_w,padding_w,borderType=cv2.BORDER_CONSTANT,value=(0,0,0))
    try:
        h_org,w_org = flat_img_org.shape[:2]
    except:
        print(flat_path)
    flat_img_resize = cv2.resize(flat_img_org, img_size)

    warp_img_org = cv2.imread(warp_path)
    warp_img_mask = warp_img_org.copy()
    # warp_img_mask[mask!=255]=0
    warp_img_mask = cv2.resize(warp_img_mask, (w_org,h_org))
    warp_img_resize = cv2.resize(warp_img_mask, img_size)
    warp_img_org = cv2.resize(warp_img_org, (w_org,h_org))

    # cv2.imshow('warp',cv2.resize(warp_img_resize,(512,512)))
    # cv2.imshow('flat',cv2.resize(flat_img_resize,(512,512)))
    # cv2.waitKey(0)
    # return 1 
    warp_img = warp_img_resize.astype(float) / 255.0
    warp_img = (warp_img-0.5)*2
    warp_img = warp_img.transpose(2, 0, 1) # NHWC -> NCHW
    warp_img = np.expand_dims(warp_img, 0)
    warp_img = torch.from_numpy(warp_img).float().cuda()

    flat_img = flat_img_resize.astype(float) / 255.0
    flat_img = (flat_img-0.5)*2
    flat_img = flat_img.transpose(2, 0, 1) # NHWC -> NCHW
    flat_img = np.expand_dims(flat_img, 0)
    flat_img = torch.from_numpy(flat_img).float().cuda()

    input = torch.cat((warp_img,flat_img),dim=1)
    # Predict
    model.eval()


    if torch.cuda.is_available():
        input = Variable(input.cuda())
    else:
        images = Variable(input)

    with torch.no_grad():
        # _,_,_,estimated_flow = model(flat_img,warp_img,F.interpolate(flat_img,(256,256)),F.interpolate(warp_img,(256,256)))
        _,_,_,_,estimated_flow = model(flat_img,warp_img,F.interpolate(flat_img,(256,256)),F.interpolate(warp_img,(256,256)))
        # bm_input=F.interpolate(pred_wc, bm_img_size)
        # outputs_bm = bm_model(bm_input)

    # pred_map = F.interpolate(pred_map,(128,128))

    # call unwarp
    ## backward
    estimated_flow = estimated_flow[-1].float()
    dewarp_im,grid= remap_using_flow_fields(warp_img_org, estimated_flow.float().squeeze()[0].cpu().numpy(),estimated_flow.float().squeeze()[1].cpu().numpy())


    ## normalize
    grid0 = grid[:,:,0]
    grid1 = grid[:,:,1]
    grid_norm = np.stack((grid0,grid1),axis=-1)
    grid_norm = np.clip(grid_norm,-1,1)
    
    grid_temp = torch.from_numpy(grid).float().cuda().unsqueeze(0)

    dewarp_im = F.grid_sample(cvimg2torch(warp_img_org).cuda(),grid_temp,padding_mode='border')
    dewarp_im = torch2cvimg(dewarp_im)[0]


    ## foreward
    flow = (estimated_flow[0].data.cpu().numpy())*5
    flow0 = flow[0]
    flow1 = flow[1]
    heat_img0 = cv2.applyColorMap(cv2.convertScaleAbs(flow0), cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
    heat_img1 = cv2.applyColorMap(cv2.convertScaleAbs(flow1), cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
    # cv2.imwrite(warp_path.replace('_capture.','_map0.'),heat_img0)
    # cv2.imwrite(warp_path.replace('_capture.','_map1.'),heat_img1)
    # distorted_im_resize, resMsk = rectification(flat_img_resize, flow)

    # pred_map = ((pred_map+1)/2*1024).astype(np.float32)
    # dewarp_im_resize = cv2.remap(warp_img_resize,pred_map[:,:,0],pred_map[:,:,1],cv2.INTER_LINEAR)
    # dewarp_im_resize = np.clip(dewarp_im_resize,0,255).astype(np.uint8)
    # dewarp_im_resize = cv2.cvtColor(dewarp_im_resize,cv2.COLOR_BGR2RGB)
   

    # show
    # if 0:
    #     # cv2.imshow('map',cv2.resize(flat_img_resize,(512,512)))
    #     cv2.imshow('flat',cv2.resize(flat_img_resize,(512,512)))
    #     cv2.imshow('warp',cv2.resize(warp_img_resize,(512,512)))
    #     cv2.imshow('dewarp',cv2.resize(dewarp_im,(512,512)))
    #     cv2.imshow('dewarp_grod',cv2.resize(dewarp_im_grid,(512,512)))
    #     cv2.waitKey(0) 

    # Save the output
    out_name = '_out_'+str(args.mode)+'.'
    cv2.imwrite(warp_path.replace('_capture.',out_name),dewarp_im)
    np.save(warp_path.replace('_capture.','_grid2.'),grid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='checkpoints/docaligner/checkpoint.pkl')
    parser.add_argument('--mode', nargs='?', type=int, default=3)
    parser.add_argument('--batchsize', nargs='?', type=int, default=4)
    parser.add_argument('--im_folder', nargs='?', type=str, default='./data/example')
    parser.set_defaults()
    args = parser.parse_args()


        
    # initial model loading
    model = get_model('docaligner', n_classes=2, in_channels=6, img_size=1024)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    if not args.model_path is None:
        checkpoint = torch.load(args.model_path,map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])


    ## inference
    capture_paths = glob.glob(os.path.join(args.im_folder,'*_capture*'))
    if args.mode==3:
        model = optimize_finetune(args,capture_paths,training_epoch=10)
        state = {'epoch': 1,
                'model_state': model.state_dict(),
                'optimizer_state' : 1,}    
        torch.save(state, args.model_path.replace('.pkl','_optimize_10epoch.pkl'))
    for im_path in tqdm(capture_paths):
        if args.mode==2:
            model = optimize(args,im_path)
        test(args,im_path,model)

