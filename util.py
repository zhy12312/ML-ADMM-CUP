# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:43:32 2022

@author: zhouhy20
"""

import torch
import math
import numpy as np
import torch.nn.functional as F

def A(x, Phi):
    return np.sum(x*Phi, axis=2) 

def At(y, Phi):
    return (Phi.transpose()*y.transpose()).transpose()

def make_embfilter(tau):
    fil = np.zeros([tau*tau,1,tau,tau])
    tij = 0
    for ti in range(tau):
        for tj in range(tau):
            fil[tij,0,ti,tj] = 1
            tij = tij+1
    return fil

# delay-embedding layer
def H(z,tau):
    pad_para_emb   = (tau-1,tau-1,tau-1,tau-1)
    pad_type_emb   = 'reflect'
    Hfil = make_embfilter(tau)
    Hfil = Hfil.astype('float32')
    Hfil=torch.tensor(Hfil).cuda().float()
    z = F.pad(z,pad_para_emb,pad_type_emb)
    return F.conv2d(z,Hfil)

# inverse delay-embedding layer
def Hinv(z,tau):
    z_size = np.array(z.shape,dtype='int32')
    Hfil = make_embfilter(tau)
    Hfil = Hfil.astype('float32')
    Hfil=torch.tensor(Hfil).cuda().float()
    Htz = F.conv_transpose2d(z,Hfil)
    return Htz[:,:,tau-1:z_size[2],tau-1:z_size[3]]/tau/tau

def psnr_block(ref, img):
    psnr = 0
    r,c,n = img.shape
    PIXEL_MAX = ref.max()
    for i in range(n):
        mse = np.mean( (ref[:,:,i] - img[:,:,i]) ** 2 )
        psnr += 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr/n

def psnr_torch(ref, img):
    psnr = 0
    nc=ref.shape[0]
    for i in range(nc):
        mse = torch.mean( (ref[i,:,:] - img[i,:,:]) ** 2 )
        psnr += 20 * torch.log10(1 / mse.sqrt())
    return psnr/nc

def shift_back(inputs,step):
    [row,col,nC] = inputs.shape
    for i in range(nC):
        inputs[:,:,i] = np.roll(inputs[:,:,i],(-1)*step*i,axis=1)
    output = inputs[:,0:col-step*(nC-1),:]
    return output

def shift(inputs,step):
    [row,col,nC] = inputs.shape
    output = np.zeros((row, col+(nC-1)*step, nC))
    for i in range(nC):
        output[:,i*step:i*step+col,i] = inputs[:,:,i]
    return output

def A_torch(x,Phi):
    x=torch.squeeze(x)
    temp = x*Phi
    y = torch.sum(temp,0)
    return y

def shift_torch(inputs, step=1):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col+(bs-1)*step)
    for i in range(bs):
        output[i, :, :, i*step:i*step+col] = inputs[i,:,:,:]
    return output.cuda()

