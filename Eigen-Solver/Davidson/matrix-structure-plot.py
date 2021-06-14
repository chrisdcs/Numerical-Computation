# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:57:51 2021

@author: m1390
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import load_sparse_matrix

A = load_sparse_matrix('usps_norm_5NN.mat')
#%%
plt.spy(A[::10,::10],marker='.',markersize=6)
plt.xticks([], [])
plt.yticks([],[])
# plt.axis('off')