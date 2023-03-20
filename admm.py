# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 20:48:58 2023

@author: zhouhy20
"""

import shutil
import time
import math
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from util import *
from mmes_denoiser import *
import scipy.io as sio



def admm_mmes(y, tau, r, sig, var, Phi, Phi_sum, mu = 0.01, rho = 0.001, denoiser=[], iter_num = 20,
                 shift_step=2, mmes_iter=[], index = None, X_ori=None, save_path = None):
    Phi_tensor = torch.from_numpy(np.transpose(Phi, (2, 0, 1))).cuda().float()
    y_tensor = torch.from_numpy(y).cuda().float() 
    truth = torch.from_numpy(np.transpose(X_ori, (2, 0, 1))).cuda().float() 
    u = At(y, Phi) 
    T = u   # default start point (initialized value)
    x = u
    v = np.zeros_like(u)
    b = np.zeros_like(T)
    begin_time = time.time()
    loss_y_min = 1
    [N1,N2,N3] = X_ori.shape
    follow1=0
    for it in range(iter_num):
        c = T + b
        yb = A(c, Phi)
        x = c + At(np.divide(y-yb, Phi_sum+mu ),Phi)
        input0 = np.random.uniform(0,1,X_ori.shape)*var        
        temp_T = shift_back(x-b, shift_step)
        follow=0
        if it==0:
            follow=1
        out, loss_y_iter = mmes_denoiser(loss_y_min,follow,follow1, input0, temp_T, truth, tau, r, sig, var, Phi_tensor, y_tensor, mmes_iter[it], mu, rho, shift_step=shift_step)   
        T = np.transpose(np.squeeze(out), (1, 2, 0))
        x_rec = T
        T = shift(T, shift_step)
        b = b-(x-T)
        mu = 0.998 * mu
        psnr_x = psnr_block(X_ori, x_rec)
        end_time = time.time()
        print('PnP-{}, Iteration {}, loss = {:.5f}, PSNR = {:.2f}dB, time = {}'.format(denoiser ,it+1, loss_y_iter, psnr_x, (end_time-begin_time)))
        if loss_y_iter < loss_y_min and it > iter_num/2+3:
            sio.savemat(save_path + 'scene0{}_{}_{:.2f}.mat'.format(index, it+1, psnr_x),{'x_rec': x_rec})
        if loss_y_iter < loss_y_min:
            loss_y_min = loss_y_iter
            shutil.move('./model_params.pth','./model_params1.pth')
            follow1=1
        if it == iter_num-1 or it == iter_num-2:
            sio.savemat(save_path + 'scene0{}_{}_{:.2f}.mat'.format(index, it+1, psnr_x),{'x_rec': x_rec})
    return x_rec