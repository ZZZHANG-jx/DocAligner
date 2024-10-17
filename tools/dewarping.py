import cv2 
import torch
import torch.nn.functional as F
import numpy as np

def torch2cvimg(tensor,min=0,max=1):
    im_list = []
    for i in range(tensor.shape[0]):
        im = tensor.detach().cpu().data.numpy()[i]
        im = im.transpose(1,2,0)
        im = np.clip(im,min,max)
        im = ((im-min)/(max-min)*255).astype(np.uint8)
        im_list.append(im)
    return im_list
def cvimg2torch(img,min=0,max=1):
    if len(img.shape)==2:
        img = np.expand_dims(img,axis=-1)
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1) # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    return img


grid3 = np.load('./example/PMC5026161_00002_grid3.jpg.npy')
im_org = cv2.imread('./example/PMC5026161_00002_origin.jpg')

grid3_temp = torch.from_numpy(grid3).float().unsqueeze(0)
grid3_temp = F.interpolate(grid3_temp.permute(0,3,1,2),im_org.shape[:2],mode='bilinear')
dewarp = torch2cvimg(F.grid_sample(cvimg2torch(im_org),grid3_temp.permute(0,2,3,1)))[0]

cv2.imwrite('visualize/dewarped.jpg',dewarp)