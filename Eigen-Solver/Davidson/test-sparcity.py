# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:08:37 2021

@author: Chris Ding
"""

import numpy as np
from utils import Davidson, plot_results, load_sparse_matrix

sparse_list = [0.0001,0.001,0.01,0.1]

mult = 1
m = 1000
n = m * mult            # Dimension of matrix

errors = [[]] * len(sparse_list)
errors_ext = [[]] * len(sparse_list)
eig_vals = []
As = []

eig = 1                 # number of eigen values to compute
l = 2                   # number of initial guess vectors
k = 4
steps = k             # number of steps
n_iters = 500
tol = 1e-10			# Convergence tolerance

V = np.zeros((n,steps*l))
# initialize guess vectors
for i in range(l*steps):
    v0 = np.random.rand(n)
    V[:,i] = v0/np.linalg.norm(v0)
    
for idx,sparsity in enumerate(sparse_list):
    a = []
    for i in range(1,m+1):
        a += mult * [i]
    A = np.diag(a) + sparsity*np.random.randn(n,n)
    A = (A.T + A)/2
    
    As.append(A)
    D = Davidson(A, eig, l, steps, n_iters, tol)
    
    

        
    restart, eigenvals, error = D.restarted_Davidson(V.copy(),True)
    errors[idx] = error[:,0]
    
    eig_vals.append(eigenvals[0])