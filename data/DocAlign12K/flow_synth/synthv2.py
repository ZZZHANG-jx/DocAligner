import cv2 
import numpy as np 
import random
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import glob
from tqdm import tqdm
import torch
import time
import os 

if __name__ =='__main__':
	image_paths = glob.glob('flat/*/*')
	random.shuffle(image_paths)
	for image_path in tqdm(image_paths):
		# if os.path.exists(image_path.replace('flat/','distorted/')):
		# 	continue
		image = cv2.imread(image_path)
		h,w = image.shape[:2]
		shape_size = image.shape[:2]
		image = cv2.resize(image,(1024,1024))
		

		###########################################################################################
		##########################################elastic distortion###############################
		### get random flow 
		alpha = (image.shape[0]+image.shape[1])
		alpha = random.uniform(0.5,2)*alpha # alpha = random.uniform(0.5,4)*alpha
		sigma = image.shape[1]*10

		## scipy smooth
		# dx = gaussian_filter((np.random.rand(*shape_size) * 2 - 1), sigma) * alpha
		# dy = gaussian_filter((np.random.rand(*shape_size) * 2 - 1), sigma) * alpha

		## opencv smooth
		# dx = (np.random.rand(*shape_size) * 2 - 1)* alpha
		# dy = (np.random.rand(*shape_size) * 2 - 1)* alpha
		# dx = cv2.GaussianBlur(dx,(0,0),31)
		# dy = cv2.GaussianBlur(dy,(0,0),31)

		## pytorch smooth (faster)
		dx = (np.random.rand(*shape_size) * 2 - 1)* alpha
		dy = (np.random.rand(*shape_size) * 2 - 1)* alpha
		def gaussian_kernel_2d(ksize, sigma):
				return cv2.getGaussianKernel(ksize, sigma) * cv2.getGaussianKernel(ksize,sigma).T
		kernel_size = 91 
		gauss_kernel = gaussian_kernel_2d(kernel_size, sigma)
		kernel = torch.from_numpy(gauss_kernel).unsqueeze(0).unsqueeze(0)
		kernel = np.repeat(kernel, 2, axis=0)
		weight = torch.nn.Parameter(data=kernel, requires_grad=False).cuda()
		dx_dy = torch.cat((torch.from_numpy(dx).unsqueeze(0).unsqueeze(0), torch.from_numpy(dy).unsqueeze(0).unsqueeze(0)),dim=1).cuda()
		dx_dy = torch.nn.functional.interpolate(dx_dy,(int(shape_size[0]/3),int(shape_size[1]/3)))
		dx_dy = torch.nn.functional.conv2d(dx_dy, weight, padding=int((kernel_size-1)/2), groups=2)
		dx_dy = torch.nn.functional.conv2d(dx_dy, weight, padding=int((kernel_size-1)/2), groups=2)
		# dx_dy = torch.nn.functional.conv2d(dx_dy, weight, padding=int((kernel_size-1)/2), groups=2)
		dx_dy = torch.nn.functional.interpolate(dx_dy,(int(shape_size[0]),int(shape_size[1])))
		dx, dy = dx_dy.cpu().squeeze(0).numpy()[0],dx_dy.cpu().squeeze(0).numpy()[1]


		### get new coordinate 
		x, y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
		## random shift
		shift_x = random.randint(-50,50)
		shift_y = random.randint(-50,50)				
		## random scale
		diff_x = x - 512
		diff_y = y - 512
		scale_x = diff_x*random.uniform(-0.05,0.2)
		scale_y = diff_y*random.uniform(-0.05,0.2)
		shift_scale_x = shift_x + scale_x
		shift_scale_y = shift_y + scale_y


		new_coordinate_save = np.concatenate((np.expand_dims(cv2.resize(y - dy - shift_scale_y,(1024,1024)),-1),np.expand_dims(cv2.resize(x - dx - shift_scale_x,(1024,1024)),-1)),-1)
		new_coordinate_save = new_coordinate_save.astype(np.float16)
		# np.save(image_path.replace('flat/','forwardmap/').replace('.jpg','.npy'),new_coordinate_save)
		
		### get distorted image
		## map_coordinates base
		# new_coordinate = np.reshape( cv2.resize(cv2.resize(y + dy + shift_scale_y,(1024,1024)),(1024,1024)), (-1, 1)), np.reshape(cv2.resize(cv2.resize(x + dx + shift_scale_x,(1024,1024)),(1024,1024)), (-1, 1))
		# synth_image = image.copy()
		# for i in range(3):
		# 	synth_image[:,:,i] = map_coordinates(image[:,:,i], new_coordinate, order=1, mode='constant',cval=255).reshape(image.shape[:2])
		# cv2.imwrite(image_path.replace('flat/','distorted/'),synth_image)
		## base on DocProj
		from utils import rectification
		new_coordinate = np.concatenate((np.expand_dims(cv2.resize(y - dy - shift_scale_y,(1024,1024)),-1),np.expand_dims(cv2.resize(x - dx - shift_scale_x,(1024,1024)),-1)),-1)	
		x, y = np.meshgrid(np.arange(1024), np.arange(1024))
		base_coordinate = np.concatenate((np.expand_dims(y,-1),np.expand_dims(x,-1)),-1)
		flow = new_coordinate - base_coordinate
		temp0 = flow[:,:,0].copy()
		temp1 = flow[:,:,1].copy()
		flow[:,:,0] = temp1
		flow[:,:,1] = temp0
		flow = np.transpose(flow,(2,0,1))
		synth_image1, resMsk = rectification(image, flow)
		# cv2.imwrite(image_path.replace('flat/','distorted/'),synth_image1)


		# cv2.imshow('synth',cv2.resize(synth_image,(512,512)))
		map0 =  cv2.applyColorMap(cv2.convertScaleAbs(temp0*2.5), cv2.COLORMAP_JET)
		map1 =  cv2.applyColorMap(cv2.convertScaleAbs(temp1*2.5), cv2.COLORMAP_JET)
		cv2.imshow('map0',cv2.resize(map0,(512,512)))
		cv2.imshow('map1',cv2.resize(map1,(512,512)))
		cv2.imshow('synth_docproj',cv2.resize(synth_image1,(512,512)))
		cv2.imshow('pdf',cv2.resize(image,(512,512)))
		cv2.waitKey(0)