# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:38:22 2022

@author: zhouhy20
"""
import torch.nn as nn
from util import *
import torch
import time

class mmes(nn.Module):
    def __init__(self, x, n_channels, ranks, N3):
        super(mmes, self).__init__()
        
        self.x = nn.Parameter(x,requires_grad=True)
        self.linear1 = nn.Linear(n_channels,ranks[0])
        self.lr1 = nn.LeakyReLU(0.2, inplace=True)
        self.linear2 = nn.Linear(ranks[0],ranks[1])
        self.lr2 = nn.LeakyReLU(0.2, inplace=True)
        self.linear3 = nn.Linear(ranks[1],ranks[2])
        self.lr3 = nn.LeakyReLU(0.2, inplace=True)
        self.linear4 = nn.Linear(ranks[2],n_channels)
        self.conv = nn.Conv2d(N3, N3, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, noise, tau, shape):
        x = self.x
        Hx = torch.transpose(torch.reshape(torch.transpose(H(x,tau),0,1),(tau*tau,shape[2]*(shape[1]+tau-1)*(shape[0]+tau-1))),0,1)
        z0 = self.linear1(Hx+noise)
        z0 = self.lr1(z0)
        z1 = self.linear2(z0)
        z1 = self.lr2(z1)
        z2 = self.linear3(z1)
        z2 = self.lr3(z2)
        AHx = self.linear4(z2)
        Xrec = Hinv(torch.transpose(torch.reshape(torch.transpose(AHx,0,1),(tau*tau,shape[2],shape[0]+tau-1,shape[1]+tau-1)),0,1),tau)
        Xrec = self.conv(torch.transpose(Xrec,0,1))
        Y = self.sig(torch.transpose(Xrec,0,1))
        return  Hx, AHx, Y
