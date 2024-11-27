import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
import torch.nn.functional as F

from tqdm import tqdm
import os
import argparse
import piqa

from models import get_model
from loaders import get_loader
from utils import dict2string,mkdir,get_lr,flow2normmap
from loss_metric import pytorch_ssim,epe

import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data.distributed import DistributedSampler 

def train(args):
    ### Log file:
    mkdir(args.logdir)
    mkdir(os.path.join(args.logdir,args.experiment_name))
    log_file_path=os.path.join(args.logdir,args.experiment_name,'log.txt')
    log_file=open(log_file_path,'a')
    log_file.write('\n---------------  '+args.experiment_name+'  ---------------\n')
    log_file.close()

    ### Setup tensorboard for visualization
    if args.tboard:
        writer = SummaryWriter(os.path.join(args.logdir,args.experiment_name,'runs'),args.experiment_name)
    dist.init_process_group(backend='nccl',init_method='env://')
    torch.cuda.set_device(args.local_rank)
    torch.cuda.manual_seed_all(42)

    ### Setup Dataloader
    data_loader = get_loader('docalign12k')
    t_loader = data_loader(args.data_path, is_transform=True, split='test',img_size=(args.img_rows, args.img_cols), augmentations=False)
    v_loader = data_loader(args.data_path, is_transform=True, split='test', img_size=(args.img_rows, args.img_cols))

    sampler_train = DistributedSampler(t_loader)
    sampler_validation = DistributedSampler(v_loader)
    trainloader = data.DataLoader(t_loader, sampler=sampler_train,batch_size=args.batch_size, num_workers=8, pin_memory=True)
    valloader = data.DataLoader(v_loader, sampler=sampler_validation,batch_size=1, num_workers=8)

    ### Setup Model
    model = get_model('docaligner')
    model.to(torch.device('cuda'))
    model=DDP(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
    ### Optimizer
    optimizer= torch.optim.Adam(model.parameters(),lr=0.0001,weight_decay=5e-4)
    ### LR Scheduler 
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    ### load checkpoint
    epoch_start=0
    if args.resume is not None:                                         
        print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume,map_location='cpu')
        # print(checkpoint['model_state'].keys())
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        # print(checkpoint['optimizer_state'])
        epoch_start=checkpoint['epoch']
        print("Loaded checkpoint '{}' (epoch {})".format(args.resume, epoch_start))

    ###-----------------------------------------Training-----------------------------------------
    ##initialize
    loss_dict = {}
    total_step = 0
    l2 = nn.MSELoss()
    l1 = nn.L1Loss()
    # l1 = nn.SmoothL1Loss()
    best = 999
    ## start training
    for epoch in range(epoch_start,args.n_epoch):
        model.train()
        for i, (warp_im,flat_im,flow) in enumerate(trainloader):
            warp_im = warp_im.float().cuda()
            flat_im = flat_im.float().cuda()
            flow = flow.float().cuda()
            pred_flow4,pred_flow3,pred_flow2,pred_flow1,pred_iter= model(flat_im,warp_im,F.interpolate(flat_im,(256,256)),F.interpolate(warp_im,(256,256)))

            loss_l1_4 = l1(pred_flow4, F.interpolate(flow,(16,16)))
            loss_l1_3 = l1(pred_flow3, F.interpolate(flow,(32,32)))
            loss_l1_2 = l1(pred_flow2, F.interpolate(flow,(128,128)))
            loss_l1_1 = l1(pred_flow1, F.interpolate(flow,(1024,1024)))

            loss_l1_iter = 0
            for iter,pred_flow_iter in enumerate(pred_iter):
                loss_l1_iter += l1(pred_flow_iter, flow) * (iter+1)*0.32
            loss = 0.01*loss_l1_4+0.02*loss_l1_3+0.08*loss_l1_2+loss_l1_iter

            pred_map1 = flow2normmap(F.interpolate(pred_iter[-1],(1024,1024),mode='bilinear'),size=1024)
            map = flow2normmap(flow,size=1024)
            loss_l1_map = l1(pred_map1, map)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_dict['l1_1']=loss_l1_1.item()
            loss_dict['l1_2']=loss_l1_2.item()
            loss_dict['l1_3']=loss_l1_3.item()
            loss_dict['l1_4']=loss_l1_4.item()
            loss_dict['l1_iter']=loss_l1_iter.item()
            loss_dict['l1_map']=loss_l1_map.item()
            loss_dict['loss']=loss.item()

            ## log
            if (i+1) % 10 == 0:
                ## print
                print('epoch[{}/{}], batch[{}/{}] -- '.format(epoch+1,args.n_epoch,i+1,len(trainloader))+dict2string(loss_dict))
                ## tbord
                if args.tboard:
                    for key,value in loss_dict.items():
                        writer.add_scalar('Train '+key+'/Iterations', value, total_step)
                ## logfile
                with open(log_file_path,'a') as f:
                    f.write('epoch[{}/{}],batch [{}/{}]--'.format(epoch+1,args.n_epoch,i+1,len(trainloader))+dict2string(loss_dict)+'\n')
        
        ## Evaluate
        psnr = piqa.PSNR()
        model.eval()
        metric_dict={'mse':0,'ssim':0,'psnr':0,'l1':0,'epe':0}
        if (epoch+1) % 1 == 0: 
            for i, (warp_im,flat_im,flow) in enumerate(tqdm(valloader)):
                warp_im = warp_im.float().cuda()
                flat_im = flat_im.float().cuda()
                flow = flow.float().cuda()
                with torch.no_grad():
                    pred_flow4,pred_flow3,pred_flow2,pred_flow1,pred_iter = model(flat_im,warp_im,F.interpolate(flat_im,(256,256)),F.interpolate(warp_im,(256,256)))

                    pred_map1 = flow2normmap(pred_iter[-1],size=1024)
                    map = flow2normmap(flow.float(),size=1024)

                    loss_l1_map = l1(pred_map1, map)
                    metric_dict['l1'] += l1(pred_map1,map).item()
                    metric_dict['epe'] += epe.shift_flow_epe(pred_map1,map).item()
                    pred_im = F.grid_sample(warp_im,pred_map1.permute(0,2,3,1))
                    gt_im = F.grid_sample(warp_im,map.permute(0,2,3,1))
                    metric_dict['mse'] += l2(pred_im,gt_im).item()
                    metric_dict['ssim'] += pytorch_ssim.ssim(pred_im,gt_im).item()
                    metric_dict['psnr'] += psnr((pred_im+1)/2,(gt_im+1)/2).item()


            ## log
            for key,value in metric_dict.items():
                metric_dict[key] = value / len(valloader)
            lrate=get_lr(optimizer)
            ## print
            print('Testing epoch {}, lr {} -- '.format(epoch+1,lrate)+dict2string(metric_dict))
            ## tbord
            if args.tboard:
                for key,value in metric_dict.items():
                    writer.add_scalar('Eval '+key+'/Epoch', value, epoch+1)
            ## logfile
            with open(log_file_path,'a') as f:
                f.write('Testing epoch {}, lr {} -- '.format(epoch+1,lrate)+dict2string(metric_dict)+'\n')

            
            if metric_dict['epe'] < best:
                best=metric_dict['epe']
                state = {'epoch': epoch+1,
                        'model_state': model.state_dict(),
                        'optimizer_state' : optimizer.state_dict(),}
                if not os.path.exists(os.path.join(args.logdir,args.experiment_name)):
                    os.system('mkdir '+ os.path.join(args.logdir,args.experiment_name))
                print('saving the best model at epoch {} with -- '.format(epoch+1)+dict2string(metric_dict))
                torch.save(state, os.path.join(args.logdir,args.experiment_name,'{}_best_'.format(epoch+1)+dict2string(metric_dict)+'.pkl').replace(', ','_').replace(' ','-'))
            sched.step(metric_dict['ssim'])

        if (epoch+1) % 10 == 0:
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            if not os.path.exists(os.path.join(args.logdir,args.experiment_name)):
                 os.system('mkdir ' + os.path.join(args.logdir,args.experiment_name))
            torch.save(state, os.path.join(args.logdir,args.experiment_name,"{}.pkl".format(epoch+1)))
        
        ## test
             


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--data_path', nargs='?', type=str, default='data/DocAlign12K/', 
                        help='Data path to load data')
    parser.add_argument('--img_rows', nargs='?', type=int, default=1024, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=1024, 
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=500, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=4, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,help='Path to previous saved model to restart from')
    parser.add_argument('--logdir', nargs='?', type=str, default='./checkpoints/docaligner',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true', 
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.add_argument('--local_rank',type=int,default=1,metavar='N')    
    parser.add_argument('--experiment_name', nargs='?', type=str,default='docaligner',
                        help='the name of this experiment')
    parser.set_defaults(tboard=True)
    args = parser.parse_args()

    train(args)