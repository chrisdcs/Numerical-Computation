# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:49:50 2021

@author: Ding Chi
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import Arnodi

# generate matrix
a = np.linspace(1,1000,1000).astype('int')
A = np.diag(a*(-1)**a)

# dim = 1000
# A = np.random.rand(dim,dim)

# compute the true eigen values
true_eig,true_eig_vec = np.linalg.eig(A)
true_eig = abs(true_eig)
true_eig_vec = abs(true_eig_vec)

# arnoldi iteration
# matrix must be square??
m = A.shape[0]

n = m
err = []

y = np.random.rand(m)
y = y/np.linalg.norm(y,2)
u = y

k = 8
tol = 1e-10
gamma = 0
iters = 100
for _ in range(iters):
    y_,lam0,lam1 = Arnodi(A, k, u)
    gamma = -abs(lam1/lam0)**_
    if np.isnan(gamma) or np.isinf(gamma):
        gamma = 0
    if abs(gamma) > 1:
        gamma = -0.1
    u = (1-gamma) * y_ + gamma * y
    y = y_
    err1 = abs(lam0 - max(true_eig))
    err.append(err1)
    if err1 < tol:
        break
    
print('lam0:',max(true_eig)), print('lam_0:',lam0)
plt.plot(np.log(err),'o-')
plt.xlabel('# of restarted Arnoldi iterations')
plt.ylabel('log error')
# plt.xticks(np.linspace(0,iters-1,iters//6).astype('int'))
plt.title('$A_1$')
# plt.title('random matrix')