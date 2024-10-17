from email.mime import base
import torch 
import numpy as np 
import torch.nn.functional as F
def shift_flow_epe(pred,gt):
    x, y = np.meshgrid(np.arange(1024), np.arange(1024))
    base_coordinate = np.expand_dims(np.stack((x,y),axis=0),axis=0)
    base_coordinate = np.tile(base_coordinate,(pred.shape[0],1,1,1))
    base_coordinate = torch.from_numpy(base_coordinate).to(pred.device)
    pred_temp = (pred+1)/2*1024 - base_coordinate
    gt_temp = (gt+1)/2*1024 - base_coordinate
    epe_org = torch.norm(pred_temp-gt_temp,p=2,dim=1).mean()

    return epe_org

def shift_flow_epe_down4(pred,gt):
    pred = F.interpolate(pred,(256,256))
    gt = F.interpolate(gt,(256,256))
    x, y = np.meshgrid(np.arange(1024), np.arange(1024))
    base_coordinate = np.expand_dims(np.stack((x,y),axis=0),axis=0)
    base_coordinate = np.tile(base_coordinate,(pred.shape[0],1,1,1))
    base_coordinate = torch.from_numpy(base_coordinate).to(pred.device).float()
    base_coordinate = F.interpolate(base_coordinate,(256,256))
    pred_temp = (pred+1)/2*1024 - base_coordinate
    gt_temp = (gt+1)/2*1024 - base_coordinate    
    epe_4 = torch.norm(pred_temp-gt_temp,p=2,dim=1).mean()
    return epe_4