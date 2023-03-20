# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 20:48:59 2023

@author: zhouhy20
"""

from models import *
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from util import *
import time
import random

def mmes_denoiser(loss_y_min,follow,follow1, x, img, truth, tau, r, sig, var, Phi_tensor, y_tensor, iter_num, mu, rho, shift_step):
    loss_min = torch.tensor([100]).cuda().float()
    [N1,N2,N3] = img.shape
    shape = [N1,N2,N3]
    img = np.expand_dims(img, 0)
    img = np.transpose(img, (3, 0, 1, 2))
    img = torch.from_numpy(img).cuda().float()
    ranks = [8*tau*tau, r, 8*tau*tau]
    x = np.expand_dims(x, 0)
    x = np.transpose(x, (3, 0, 1, 2))
    x = torch.from_numpy(x).cuda().float()
    model = mmes(x, tau*tau, ranks, N3).cuda()
    if follow1:
        model.load_state_dict(torch.load('model1.pth'))
    lam=5
    optimizer = optim.Adam(model.parameters(), lr = 0.0125, betas=(0.9, 0.999))
    loss_fn = nn.MSELoss().cuda()
    cost_hist = np.zeros([iter_num])
    i=0
    j=0
    while i<iter_num:
        noise = torch.normal(0, sig, [N3*(N2+tau-1)*(N1+tau-1), tau*tau]).cuda().float()
        Hx, AHx, out= model(noise, tau, shape)
        optimizer.zero_grad()
        outshift = shift_torch(out, shift_step)
        y_est = A_torch(outshift, Phi_tensor)
        y_loss = loss_fn(y_est, y_tensor) *rho/mu
        x_loss = loss_fn(img, out)
        ae_loss = loss_fn(Hx, AHx)*lam
        cost_hist[i] = y_loss
        if follow:
            loss = y_loss + ae_loss
            loss.backward()
            optimizer.step()
            if i %10 == 0:
                if y_loss < ae_loss/lam:
                    lam = lam*1.1
                elif ae_loss/lam < y_loss:
                    lam = lam*0.99
        else:
            loss = y_loss + ae_loss + x_loss
            loss.backward()
            optimizer.step()
            
            if i %10 == 0:
                if y_loss < ae_loss/lam:
                    lam = lam*1.01
                elif ae_loss/lam < y_loss:
                    lam = lam*0.99
        if (i+1)%25==0 and y_loss < loss_min*1.1:
            loss_min = y_loss
            output = out.detach().cpu().numpy()
            torch.save(model.state_dict(), 'model.pth')
        if (i+1)%100==0:
            PSNR = psnr_torch(truth, torch.squeeze(out))
            print('ML iter {}, x_loss:{:.5f}, y_loss:{:.5f},ae_loss:{:.5f},lam:{:.5f},PSNR:{:.2f}'.format(i+1+j*200, x_loss.detach().cpu().numpy(), y_loss.detach().cpu().numpy(),ae_loss.detach().cpu().numpy(),lam, PSNR.detach().cpu().numpy()))
        if i==iter_num-1 and loss_min.detach().cpu().numpy() > loss_y_min:
            i=i-500
            j += 1
        i += 1
    return output, loss_min.detach().cpu().numpy()