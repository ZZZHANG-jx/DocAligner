import torch 

def Smoothloss(output):
    '''
    output -> (N,C,H,W)
    '''
    dy = torch.abs(output[:,:,1:,:]-output[:,:,:-1,:])
    # dy[:,:,50:-50,:] = 0
    dx = torch.abs(output[:,:,:,1:]-output[:,:,:,:-1])
    # dx[:,:,:,50:-50] = 0
    dx = torch.mul(dx,dx)
    dy = torch.mul(dy,dy)
    d = torch.mean(dx)+torch.mean(dy)
    return d/2.

def Smoothlossv2(flow):
    pred_flow1_padding = flow.clone()
    # pred_flow1_padding[:,:,100:-100,100:-100] = 0
    regularization_loss = torch.nn.functional.l1_loss(pred_flow1_padding,torch.zeros_like(pred_flow1_padding))
    return regularization_loss

def TVloss(x):
    '''
    x -> (N,C,H,W)
    '''
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h =  (x.size()[2]-1) * x.size()[3]
    count_w = x.size()[2] * (x.size()[3] - 1)
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)/batch_size