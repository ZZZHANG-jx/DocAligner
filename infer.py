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
def self_augment_batch(img, grid_paths, num):
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
    return remapped_image,grid

def optimize_singleim(args,warp_path):

    BATCHSIZE=args.batchsize
    ttt = time.time()
    if not args.model_path is None:
        model.load_state_dict(checkpoint['model_state'])
    print(time.time()-ttt)
    optimizer= torch.optim.Adam(model.parameters(),lr=1e-4)


    model.train()
    for iteration in range(50):
        img_size=(1024,1024)
        flat_path = warp_path.replace('_capture.','_target.')
        flat_img_org = cv2.imread(flat_path)
        h,w = flat_img_org.shape[:2]
        h_org,w_org = flat_img_org.shape[:2]
        flat_img_resize = cv2.resize(flat_img_org, img_size)

        warp_img_org = cv2.imread(warp_path)
        warp_img_resize = cv2.resize(warp_img_org, img_size)
        warp_img_org = cv2.resize(warp_img_org, (w_org,h_org))

        warp_img_list = self_augment_batch(warp_img_resize,glob.glob('./data/DocAlign12K/forwardmap/*'),BATCHSIZE-1)

        warp_sobels = torch.zeros(BATCHSIZE,1,1024,1024).float().cuda()
        for i, warp_img_resize in enumerate(warp_img_list):
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

        l1 = nn.L1Loss()
        _,_,_,_,pred_iter = model(flat_imgs,warp_imgs,F.interpolate(flat_imgs,(256,256)),F.interpolate(warp_imgs,(256,256)))

        loss = 0
        num_iter = len(pred_iter)
        for iter in range(num_iter):
            pred_map1 = flow2normmap(F.interpolate(pred_iter[iter],(1024,1024)),size=1024)
            pred_map1_0_max,pred_map1_1_max = torch.max(torch.max(pred_map1[:,0,:,:],dim=-1)[0],dim=-1)[0],torch.max(torch.max(pred_map1[:,1,:,:],dim=-1)[0],dim=-1)[0]
            pred_map1_0_min,pred_map1_1_min = torch.min(torch.min(pred_map1[:,0,:,:],dim=-1)[0],dim=-1)[0],torch.min(torch.min(pred_map1[:,1,:,:],dim=-1)[0],dim=-1)[0]
            max_gt = torch.ones_like(pred_map1_0_max)
            min_gt = torch.ones_like(pred_map1_0_min)*(-1)
            range_loss = l1(pred_map1_0_min,min_gt) + l1(pred_map1_1_min,min_gt) + l1(pred_map1_0_max,max_gt) + l1(pred_map1_1_max,max_gt)
            smooth_loss = smooth.Smoothlossv2(pred_iter[iter]) 
            pred_sobels = F.grid_sample(warp_sobels.float(),pred_map1.permute(0,2,3,1).float())
            loss_l1 = l1(pred_sobels,flat_sobels)
            loss += (loss_l1*500 + range_loss*10 + smooth_loss*0.001 + smooth.TVloss(pred_iter[iter])*5) * (0.8**(num_iter - iter))
            
        loss = loss
        metric = l1(pred_sobels[-1].detach(),flat_sobels[-1].detach()).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('iter {}, metric {:.4f} loss {:.4f}'.format(iteration+1,metric,loss.item()))

    return model

def optimize_multiim(args,warp_paths,training_epoch=10):
    optimizer= torch.optim.Adam(model.parameters(),lr=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.2,step_size=100)        

    img_size=(1024,1024)
    BATCHSIZE=args.batchsize
    model.train()
    l1 = nn.L1Loss()
    if len(warp_paths) < BATCHSIZE:
        warp_paths = (warp_paths)*(BATCHSIZE+1)
    
    for epoch in range(training_epoch):
        metric_dict = {'l1':0.}
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
                warp_img_resize = self_augment(warp_img_resize,glob.glob('./data/DocAlign12K/forwardmap/*'))
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
            _,_,_,_,pred_iter = model(flat_imgs,warp_imgs,F.interpolate(flat_imgs,(256,256)),F.interpolate(warp_imgs,(256,256)))
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
                
                ## visualize
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
            metric_dict['l1'] += metric
            print('epoch {}, iter {}, metric {:.4f} loss {:.4f}'.format(epoch+1, iter+1,metric,loss.item()))
        # print(metric_dict['l1']/iter_num)
        sched.step()

    return model



def test(args,warp_path,model):

    img_size=(1024,1024)

    ## Setup image
    flat_path = warp_path.replace('_capture.','_target.')
    flat_img_org = cv2.imread(flat_path)
    h_org,w_org = flat_img_org.shape[:2]
    flat_img_resize = cv2.resize(flat_img_org, img_size)

    warp_img_org = cv2.imread(warp_path)
    warp_img_resize = cv2.resize(warp_img_org, img_size)
    warp_img_org = cv2.resize(warp_img_org, (w_org,h_org))

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
    input = Variable(input.cuda())

    ## Predict
    model.eval()
    with torch.no_grad():
        _,_,_,_,estimated_flow = model(flat_img,warp_img,F.interpolate(flat_img,(256,256)),F.interpolate(warp_img,(256,256)))


    ## dewarping
    # get grid
    estimated_flow = estimated_flow[-1].float()
    dewarp_im,grid= remap_using_flow_fields(warp_img_org, estimated_flow.float().squeeze()[0].cpu().numpy(),estimated_flow.float().squeeze()[1].cpu().numpy())
    # normalize
    grid0 = grid[:,:,0]
    grid1 = grid[:,:,1]
    grid_norm = np.stack((grid0,grid1),axis=-1)
    grid_norm = np.clip(grid_norm,-1,1)
    grid_temp = torch.from_numpy(grid).float().cuda().unsqueeze(0)
    dewarp_im = F.grid_sample(cvimg2torch(warp_img_org).cuda(),grid_temp,padding_mode='border')
    dewarp_im = torch2cvimg(dewarp_im)[0]

    ## visualize
    # flow = (estimated_flow[0].data.cpu().numpy())*5
    # flow0 = flow[0]
    # flow1 = flow[1]
    # heat_img0 = cv2.applyColorMap(cv2.convertScaleAbs(flow0), cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
    # heat_img1 = cv2.applyColorMap(cv2.convertScaleAbs(flow1), cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
    # cv2.imshow('map',cv2.resize(flat_img_resize,(512,512)))
    # cv2.imshow('flat',cv2.resize(flat_img_resize,(512,512)))
    # cv2.imshow('warp',cv2.resize(warp_img_resize,(512,512)))
    # cv2.imshow('dewarp',cv2.resize(dewarp_im,(512,512)))
    # cv2.waitKey(0) 

    # Save the output
    out_name = '_out_'+str(args.mode)+'.'
    cv2.imwrite(warp_path.replace('_capture.',out_name),dewarp_im)
    np.save(warp_path.replace('_capture.','_grid2.'),grid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='checkpoint/docaligner.pkl')
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
    capture_paths = glob.glob(os.path.join(args.im_folder,'*_capture.*'))
    if args.mode==1:
        for im_path in tqdm(capture_paths):
            test(args,im_path,model)
    elif args.mode==2:
        for im_path in tqdm(capture_paths):
            model = optimize_singleim(args,im_path)
            test(args,im_path,model)
    elif args.mode==3:
        model = optimize_multiim(args,capture_paths,training_epoch=10)
        for im_path in tqdm(capture_paths):
            test(args,im_path,model)
    else:
        raise ValueError('mode need to be 1, 2 or 3')

