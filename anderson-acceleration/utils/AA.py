# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 17:02:55 2021

@author: Chi Ding
"""

import numpy as np

class AA:
    def __init__(self,m):
        # m is the depth
        self.m = m
        self.mk = 0
        
        # memory store the historic residual
        self.memory = []
        
        self.step = 1#0
        
    def LS(self):
        # solve coefficients for AA
        gamma = None
        
        f = np.array(self.memory)[-self.mk:]
        f_ = np.array(self.memory)[-self.mk-1:-1]
        F = f - f_
        
        
        return gamma
    
    def iterate(self,x,gx):
        
        self.memory.append(gx - x)
        self.step += 1
        self.mk = min(self.m, self.step)
        
        return