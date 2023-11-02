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
    Z = At(y, Phi)   
    I = At(y, Phi)
    b = np.zeros_like(Z)
    begin_time = time.time()
    loss_y_min = 1
    [N1,N2,N3] = X_ori.shape
    follow1=0
    num=0

    for it in range(iter_num):   
        c = Z + b
        yb = A(c, Phi)
        I = c + At(np.divide(y-yb, Phi_sum+mu ),Phi)
        input0 = np.random.uniform(0,1,X_ori.shape)*var        
        temp_Z = shift_back(I-b, shift_step)
        follow=0
        if it==0:
            follow=1
        out, loss_y_iter = mmes_denoiser(loss_y_min,follow,follow1, input0, temp_Z, truth, tau, r, sig, var, Phi_tensor, y_tensor, mmes_iter[it], mu, rho, shift_step=shift_step)   
        if it > 0:
           r0 = I - shift(np.transpose(np.squeeze(out), (1, 2, 0)), shift_step)
           s = mu * (shift(np.transpose(np.squeeze(out), (1, 2, 0)), shift_step) - Z)
           a = np.linalg.norm(r0)/np.linalg.norm(s)
           if a > 10:
               mu = 2 * mu
           if a < 0.1:
               mu = mu * 0.5
        Z = np.transpose(np.squeeze(out), (1, 2, 0))
        x_rec = Z
        Z = shift(Z, shift_step)
        b = b-(I-Z) 
        #mu = 0.998 * mu
        psnr_x = psnr_block(X_ori, x_rec)
        end_time = time.time()
        print('ADMM-{}, Iteration {}, loss = {:.5f}, PSNR = {:.2f}dB, time = {}'.format(denoiser ,it+1, loss_y_iter, psnr_x, (end_time-begin_time)))
        if loss_y_iter < loss_y_min and it > iter_num/2+3:
            sio.savemat(save_path + 'scene0{}_{}_{:.2f}.mat'.format(index, it+1, psnr_x),{'x_rec': x_rec})
        if loss_y_iter < loss_y_min:
            loss_y_min = loss_y_iter
            shutil.move('./model.pth','./model1.pth')
            follow1=1
        if it == iter_num-1 or it == iter_num-2:
            sio.savemat(save_path + 'scene0{}_{}_{:.2f}.mat'.format(index, it+1, psnr_x),{'x_rec': x_rec})
    return x_rec