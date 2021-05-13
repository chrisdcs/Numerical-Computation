# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:49:50 2021

@author: Ding Chi
"""

import numpy as np
from utils import Arnoldi, plot_results, load_sparse_matrix

# generate matrix
# a = np.linspace(1,1000,1000).astype('int')
# A = np.diag(a*(-1)**a)

# A = load_sparse_matrix('ss1.mat')

A = load_sparse_matrix('ifiss_mat.mat')

m = A.shape[0]

y = np.random.rand(m)
y = y/np.linalg.norm(y,2)
#%% Parameters
k = 8
tol = 1e-10
gamma = 0
iters = 200
extrapolate = True
#%% 
A_ext = Arnoldi(A, k, tol)
Arn = Arnoldi(A, k, tol)
A_ext.restarted_arnoldi(y, extrapolate, iters)#, true_eigenval=max(true_eig))
Arn.restarted_arnoldi(y, not extrapolate, iters)#, true_eigenval=max(true_eig))
#%%
label_ext = '8-step $γ = γ_s$'
label = '8-step'
title = 'A3'
plot_results(A_ext, label_ext, Arn, label, title)