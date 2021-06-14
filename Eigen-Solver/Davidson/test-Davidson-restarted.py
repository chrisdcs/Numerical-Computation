# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:08:10 2021

@author: m1390
"""

import numpy as np
import scipy.sparse.linalg as LA
import scipy.sparse as sparse
from utils import Davidson, plot_results, load_sparse_matrix
import time


tol = 1e-10			# Convergence tolerance
sparsity = 0.01

a = []
mult = 5
m = 1000
n = m * mult            # Dimension of matrix
for i in range(1,m+1):
    a += mult * [i]
A = np.diag(a) + sparsity*np.random.randn(n,n)
A = (A.T + A)/2

# uncomment the following and comment the previous matrix A to work with sparse matrices
# A = load_sparse_matrix('TEM27623.mat')
# n=A.shape[0]


# I = np.eye(n)
# k step Lanczos iteration

eig = 5                 # number of eigen values to compute
l = 6                   # number of initial guess vectors: could be larger than 1 for each eigenvalue
k = 8                   # k-step Davidson
steps = k               # number of steps
n_iters = 100
D = Davidson(A, eig, l, steps, n_iters, tol)

V = np.zeros((n,steps*l))

# initialize guess vectors and collect them as V
for i in range(l*steps):
    v0 = np.random.rand(n)
    V[:,i] = v0/np.linalg.norm(v0)

# V = np.eye(n,l*steps)

# number of initial guess must be larger or equal to number of eigen values we are trying to solve
if l < eig: raise Exception('l must be >= number of eigenvalues')

start_davidson = time.time()


"""
    Initialization matters, if we use random vectors, converges very slowly
"""
restart, eigenvals, errors = D.restarted_Davidson(V.copy(),True)
end_davidson = time.time()


restart_,eigenvals_,errors_ = D.restarted_Davidson(V.copy())



print("davidson = ", eigenvals[:eig],";",
    end_davidson - start_davidson, "seconds")
#%% Compare with package solvers
# End of Numpy/scipy diagonalization. Print results.
start_numpy = time.time()
if sparse.issparse(A):
    E,Vec = LA.eigs(A)#,k=1,which='SM')
    E = np.sort(E)
    end_numpy = time.time()
    print("scipy = ", E[:eig],";",
      end_numpy - start_numpy, "seconds")
else:
    E,Vec = np.linalg.eig(A)
    E = np.sort(E)
    end_numpy = time.time()
    print("numpy = ", E[:eig],";",
      end_numpy - start_numpy, "seconds")

#%%
for i in range(eig):
    err_ext = np.array(errors[:,i])
    label_ext = str(steps)+'-step $γ = -0.5$'
    # title = 'eigenvalue' + str(i+1)
    title = None
    
    err = np.array(errors_[:,i])
    label = str(steps)+'-step'
    
    common = max(err_ext.min(), err.min())
    err_ext = err_ext[err_ext >= common]
    err = err[err >= common]
    
    plot_results(err_ext, label_ext, err, label, title)