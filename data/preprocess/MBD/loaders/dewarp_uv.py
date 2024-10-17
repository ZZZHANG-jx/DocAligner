import os
import argparse
import matplotlib.pyplot as plt
import hdf5storage as h5
import cv2
import random
import subprocess
import glob
import numpy as np
import scipy.spatial.qhull as qhull

def interp_weights(xyz, uvw):
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    # pixel_triangle = pixel[tri.simplices]
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, 2]
    bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('njk,nj->nk', np.take(values, vtx, axis=0), wts)

def dewarp_base_displacement(displacement_map,image,is_shift):
    '''
    displacement -> ndarray, 
    image -> ndarray, uint8
    is_shift -> shift (displacement -h~h -w~w) or new_coordinates (displacement 0~h 0~w)
    '''
    shift=displacement_map
    flat_shape = shift.shape[:2]
    perturbed_label_classify = np.ones(flat_shape).reshape(flat_shape[0] * flat_shape[1])
    flat_img = np.full_like(image,256,dtype=np.uint16)
    origin_pixel_position = np.argwhere(np.zeros(flat_shape, dtype=np.uint32) == 0).reshape(flat_shape[0] * flat_shape[1], 2)
    perturbed_label = shift.reshape(flat_shape[0] * flat_shape[1], 2)
    perturbed_img = image.reshape(flat_shape[0] * flat_shape[1], 3)

    if is_shift:
        perturbed_label += origin_pixel_position
    else:
        pass
    pixel_position = perturbed_label
    pixel = perturbed_img
    '''construct Delaunay triangulations in all scattered pixels and then using interpolation'''
    ### pixel_position 新坐标， origin_pixel_position 原基本坐标
    vtx, wts = interp_weights(pixel_position, origin_pixel_position)
    # for i in range(pixel_position.shape[0]):
    # wts[np.abs(wts)>=1]=0
    wts_sum = np.abs(wts).sum(-1)
    wts = wts[wts_sum <= 1, :]
    vtx = vtx[wts_sum <= 1, :]
    flat_img.reshape(flat_shape[0] * flat_shape[1], 3)[wts_sum <= 1, :] = interpolate(pixel, vtx, wts)
    flat_img = flat_img.reshape(flat_shape[0], flat_shape[1], 3)
    flat_img = flat_img.astype(np.uint8)
    # cv2.imshow('dewarp',flat_img)
    # cv2.imshow('image',image)
    # cv2.imshow('uv',uv)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return flat_img


if __name__ == '__main__':
    image_paths = glob.glob('./dataset_demo/img/*/*')
    for image_path in image_paths:
        image = cv2.imread(image_path)
        uv_path = image_path.replace('.png','.exr').replace('img','uv')
        uv=cv2.imread(uv_path,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        uv[:,:,1] = (np.ones_like(uv[:,:,1]) - uv[:,:,1]).copy()*uv[:,:,0]
        shift=uv[:,:,1:]*448
        flat_image = dewarp_base_displacement(shift,image,False)
