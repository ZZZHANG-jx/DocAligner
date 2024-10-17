'''
Misc Utility functions
'''
from collections import OrderedDict
from email.mime import base
import os
import numpy as np
import torch
import random
import torchvision
import torch.nn as nn
import cv2
import torch.nn.functional as F
class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
# 
    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        x = (x-x.min())/(x.max()-x.min()+ 1e-6)
        x = (x-0.5)*2        
        return x

class GradLoss(nn.Module):

    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad)

def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D
def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D
def get_thin_kernels(start=0, end=360, step=45):
        k_thin = 3  # actual size of the directional kernel
        # increase for a while to avoid interpolation when rotating
        k_increased = k_thin + 2

        # get 0° angle directional kernel
        thin_kernel_0 = np.zeros((k_increased, k_increased))
        thin_kernel_0[k_increased // 2, k_increased // 2] = 1
        thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

        # rotate the 0° angle directional kernel to get the other ones
        thin_kernels = []
        for angle in range(start, end, step):
            (h, w) = thin_kernel_0.shape
            # get the center to not rotate around the (0, 0) coord point
            center = (w // 2, h // 2)
            # apply rotation
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

            # get the k=3 kerne
            kernel_angle = kernel_angle_increased[1:-1, 1:-1]
            is_diag = (abs(kernel_angle) == 1)      # because of the interpolation
            kernel_angle = kernel_angle * is_diag   # because of the interpolation
            thin_kernels.append(kernel_angle)
        return thin_kernels

class BlurFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=True):
        super(BlurFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'

        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        self.gaussian_filter.weight.requires_grad=False
        self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D).to('cuda')
    def forward(self, img):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to('cuda')

        # gaussian
        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])
        return blurred

class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=True):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'

        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        self.gaussian_filter.weight.requires_grad=False
        self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)

        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_x.weight.requires_grad = False
        self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)


        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_y.weight.requires_grad= False
        self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)


        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        self.directional_filter.weight.requires_grad=False
        self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.hysteresis.weight.requires_grad=False
        self.hysteresis.weight[:] = torch.from_numpy(hysteresis)


    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian

        for c in range(C):
            # blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])
            blurred[:, c:c+1] = img[:, c:c+1]

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])

        # thick edges

        grad_x, grad_y = grad_x, grad_y
        grad_magnitude = (grad_x ** 2 + grad_y ** 2)# ** 0.5
        grad_magnitude = grad_magnitude/(grad_magnitude.max()-grad_magnitude.min())
        grad_magnitude = (grad_magnitude-0.5)*2
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # # thin edges

        # directional = self.directional_filter(grad_magnitude)
        # # get indices of positive and negative directions
        # positive_idx = (grad_orientation / 45) % 8
        # negative_idx = ((grad_orientation / 45) + 4) % 8
        # thin_edges = grad_magnitude.clone()
        # # non maximum suppression direction by direction
        # for pos_i in range(4):
        #     neg_i = pos_i + 4
        #     # get the oriented grad for the angle
        #     is_oriented_i = (positive_idx == pos_i) * 1
        #     is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
        #     pos_directional = directional[:, pos_i]
        #     neg_directional = directional[:, neg_i]
        #     selected_direction = torch.stack([pos_directional, neg_directional])

        #     # get the local maximum pixels for the angle
        #     is_max = selected_direction.min(dim=0)[0] > 0.0
        #     is_max = torch.unsqueeze(is_max, dim=1)

        #     # apply non maximum suppression
        #     to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
        #     thin_edges[to_remove] = 0.0

        # # thresholds

        # if low_threshold is not None:
        #     low = thin_edges > low_threshold

        #     if high_threshold is not None:
        #         high = thin_edges > high_threshold
        #         # get black/gray/white only
        #         thin_edges = low * 0.5 + high * 0.5

        #         if hysteresis:
        #             # get weaks and check if they are high or not
        #             weak = (thin_edges == 0.5) * 1
        #             weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
        #             thin_edges = high * 1 + weak_is_high * 1
        #     else:
        #         thin_edges = low * 1
        return grad_magnitude
        # return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges

def scale_invariant_loss(output, target):
    d = output-target
    return torch.abs(d).mean() + torch.abs(d.mean())
def smooth_loss(output):
    dy = torch.abs(output[:,:,1:,:]-output[:,:,:-1,:])
    dx = torch.abs(output[:,:,:,1:]-output[:,:,:,:-1])
    dx = torch.mul(dx,dx)
    dy = torch.mul(dy,dy)
    d = torch.mean(dx)+torch.mean(dy)
    return d/2.
    
def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9,):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*(1 - iter/max_iter)**power


def adjust_learning_rate(optimizer, init_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images 
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value

    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])

def visualize(epoch,model,layer):    
    #get conv layers
    conv_layers=[]
    for m in model.modules():
        if isinstance(m,torch.nn.modules.conv.Conv2d):
            conv_layers.append(m)

    # print conv_layers[layer].weight.data.cpu().numpy().shape
    tensor=conv_layers[layer].weight.data.cpu()
    vistensor(tensor, epoch, ch=0, allkernels=False, nrow=8, padding=1)


def vistensor(tensor, epoch, ch=0, allkernels=False, nrow=8, padding=1): 
    '''
        vistensor: visuzlization tensor
        @ch: visualization channel 
        @allkernels: visualization all tensors
        https://github.com/pedrodiamel/pytorchvision/blob/a14672fe4b07995e99f8af755de875daf8aababb/pytvision/visualization.py#L325
    ''' 
    
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c,-1,w,h )
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
        
    rows = np.min( (tensor.shape[0]//nrow + 1, 64 )  )
    # print rows
    # print tensor.shape
    grid = utils.make_grid(tensor, nrow=8, normalize=True, padding=padding)
    # print grid.shape
    plt.figure( figsize=(10,10), dpi=200 )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.savefig('./generated/filters_layer1_dwuv_'+str(epoch)+'.png')
    plt.close()


def show_uloss(uwpred,uworg,inp_img, samples=7):
    
    n,c,h,w=inp_img.shape
    # print(labels.shape)
    uwpred=uwpred.detach().cpu().numpy()
    uworg=uworg.detach().cpu().numpy()
    inp_img=inp_img.detach().cpu().numpy()

    #NCHW->NHWC
    uwpred=uwpred.transpose((0, 2, 3, 1))
    uworg=uworg.transpose((0, 2, 3, 1))

    choices=random.sample(range(n), min(n,samples))
    f, axarr = plt.subplots(samples, 3)
    for j in range(samples):
        # print(np.min(labels[j]))
        # print imgs[j].shape
        img=inp_img[j].transpose(1,2,0)
        axarr[j][0].imshow(img[:,:,::-1])
        axarr[j][1].imshow(uworg[j])
        axarr[j][2].imshow(uwpred[j])
    
    plt.savefig('./generated/unwarp.png')
    plt.close()


def show_uloss_visdom(vis,uwpred,uworg,labels_win,out_win,labelopts,outopts,args):
    samples=7
    n,c,h,w=uwpred.shape
    uwpred=uwpred.detach().cpu().numpy()
    uworg=uworg.detach().cpu().numpy()
    out_arr=np.full((samples,3,args.img_rows,args.img_cols),0.0)
    label_arr=np.full((samples,3,args.img_rows,args.img_cols),0.0)
    choices=random.sample(range(n), min(n,samples))
    idx=0
    for c in choices:
        out_arr[idx,:,:,:]=uwpred[c]
        label_arr[idx,:,:,:]=uworg[c]
        idx+=1

    vis.images(out_arr,
               win=out_win,
               opts=outopts)
    vis.images(label_arr,
               win=labels_win,
               opts=labelopts)

def show_unwarp_tnsboard(global_step,writer,uwpred,uworg,grid_samples,gt_tag,pred_tag):
    idxs=torch.LongTensor(random.sample(range(images.shape[0]), min(grid_samples,images.shape[0])))
    grid_uworg = torchvision.utils.make_grid(uworg[idxs],normalize=True, scale_each=True)
    writer.add_image(gt_tag, grid_uworg, global_step)
    grid_uwpr = torchvision.utils.make_grid(uwpred[idxs],normalize=True, scale_each=True)
    writer.add_image(pred_tag, grid_uwpr, global_step)

def show_wc_tnsboard(global_step,writer,images,labels, pred, grid_samples,inp_tag, gt_tag, pred_tag):
    idxs=torch.LongTensor(random.sample(range(images.shape[0]), min(grid_samples,images.shape[0])))
    grid_inp = torchvision.utils.make_grid(images[idxs],normalize=True, scale_each=True)
    writer.add_image(inp_tag, grid_inp, global_step)
    grid_lbl = torchvision.utils.make_grid(labels[idxs],normalize=True, scale_each=True)
    writer.add_image(gt_tag, grid_lbl, global_step)
    grid_pred = torchvision.utils.make_grid(pred[idxs],normalize=True, scale_each=True)
    writer.add_image(pred_tag, grid_pred, global_step)
def dict2string(loss_dict):
    loss_string = ''
    for key, value in loss_dict.items():
        loss_string += key+' {:.4f}, '.format(value)
    return loss_string[:-2]
def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)    

def flow2normmap(flow,size=1024):
    bs,n,w,h = flow.shape
    x, y = np.meshgrid(np.arange(size),np.arange(size))
    base_cord = np.stack((x,y),axis=0)
    base_cord = np.tile(np.expand_dims(base_cord,axis=0),(bs,1,1,1))
    base_cord = torch.from_numpy(base_cord).to(flow.device)
    map = ((flow+base_cord)/size-0.5)*2
    return map
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
    if len(img.shape)==2:
        img = np.expand_dims(img,axis=-1)
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1) # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    return img

def get_sobel(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0)  
    y = cv2.Sobel(img,cv2.CV_16S,0,1)  
    absX = cv2.convertScaleAbs(x)   # 转回uint8  
    absY = cv2.convertScaleAbs(y)  
    high_frequency = cv2.addWeighted(absX,0.5,absY,0.5,0)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_BGR2GRAY)
    return high_frequency