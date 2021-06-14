# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:54:38 2021

@author: m1390
"""

import numpy as np
# from utils import load_sparse_matrix

# generate a random symmetric matrix
# n = 100
# A = np.random.rand(n,n)
# A = A + A.T

A = np.diag(np.concatenate([np.linspace(0,2,201),[2.5,3]]))
n = A.shape[0]

# k step Lanczos iteration
k = 20
b = np.random.rand(n)
b = b/np.linalg.norm(b)

beta = 0

T = np.zeros((k, k))
Q = np.zeros((n, k+1))

Q[:,0] = b

for i in range(k):
    
    v = A @ Q[:,i]
        
    alpha = Q[:,i] @ v
    T[i,i] = alpha
    
    if i > 0:
        v = v - beta * Q[:,i-1] - alpha * Q[:,i]
    else:
        v = v - alpha * Q[:,i]
    
    beta = np.linalg.norm(v)
    
    Q[:,i+1] = v/beta
    
    if i < k - 1:
        T[i,i+1] = beta
        T[i+1,i] = beta
    
u_,v_ = np.linalg.eig(T)
u,v = np.linalg.eig(A)