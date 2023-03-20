# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:51:47 2022

@author: zhouhy20
"""

import os
import time
import math
import numpy as np
from numpy import *
import scipy.io as sio
from statistics import mean
from admm import *
from util import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sample = '1'
save_path = './Result/result'+ sample + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
mydata='./data/scene1.mat'
shift_step = 2
file = sio.loadmat(mydata) 
y = np.float32(file['meas'])
Phi = np.float32(file['mask'])
Phi_sum = np.sum(Phi**2,2)
Phi_sum[Phi_sum==0]=1
orig = np.float32(file['orig'])
X_ori=shift_back(orig,shift_step)
index = int(sample)
mu = 0.5
denoiser = 'mmes'
iter_num = 21
mmes_iter = [8000]+[500]*20
rho = 0.5
r=16
tau=4
sig=0.05
var=1
x_rec = admm_mmes(y, tau, r, sig, var, Phi, Phi_sum, mu=mu, rho=rho,
                     denoiser=denoiser, iter_num=iter_num,shift_step=shift_step,  
                     mmes_iter=mmes_iter, index = index, X_ori=X_ori, save_path = save_path)
