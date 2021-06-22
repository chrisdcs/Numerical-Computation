# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:08:10 2021

@author: Chris Ding
"""

import numpy as np
import scipy.sparse.linalg as LA
import scipy.sparse as sparse
from utils import Davidson, plot_results, load_sparse_matrix
import time
import argparse
#%% set parameters

tol = 1e-8			        # Convergence tolerance
data_file_name = r'data/rhfHBAR.npz'
n_eig = 10                  # number of eigen values to compute
n_guess = 15                # number of initial guess vectors: could be larger than 1 for each eigenvalue
k = 8                       # k-step Davidson
steps = k                   # number of steps
max_iter = 200              # max number of times to run restarted Davidson
descent_order = False       # if descent order True, we are solving max eigenvalues, vice versa
init = 'Euclidean'          # type of guess vector initialization

#%% initialization and sanity check
# number of initial guess must be larger or equal to number of eigenvalues
if n_guess < n_eig: raise Exception(
        'number of guess vectors must be >= number of eigenvalues')

A = load_sparse_matrix(data_file_name)
n = A.shape[0]

# initialize guess vectors and collect them as V
V = np.zeros((n,steps*n_guess))
if init == 'random':
    for i in range(n_guess):
        v0 = np.random.rand(n)
        V[:,i] = v0/np.linalg.norm(v0)
elif init == 'Euclidean':
    # Standard Euclidean basis
    V[:,:n_guess] = np.eye(n,n_guess)
    
#%% Initialize algorithm and computation

D = Davidson(A, n_eig, n_guess, steps, max_iter, tol, descent = descent_order)
D_ = Davidson(A, n_eig, n_guess, steps, max_iter, tol, descent = descent_order)

start_davidson = time.time()


"""
    Initialization matters, if we use random vectors, converges very slowly
"""
restart, eigenvals, errors = D.restarted_Davidson(V.copy(),True)
end_davidson = time.time()


restart_,eigenvals_,errors_ = D_.restarted_Davidson(V.copy())

print("davidson = ", eigenvals[:n_eig],";",
    end_davidson - start_davidson, "seconds")
#%% Compare with package solvers
# End of Numpy/scipy diagonalization. Print results.
start_numpy = time.time()
if sparse.issparse(A):
    E,Vec = LA.eigs(A)#,k=1,which='SM')
    E = np.sort(E)
    end_numpy = time.time()
    print("scipy = ", E[:n_eig],";",
      end_numpy - start_numpy, "seconds")
else:
    E,Vec = np.linalg.eig(A)
    E = np.sort(E)
    end_numpy = time.time()
    print("numpy = ", E[:n_eig],";",
      end_numpy - start_numpy, "seconds")

#%%
for i in range(n_eig):
    err_ext = np.array(errors[:,i])
    label_ext = str(steps)+'-step $γ = -0.5$'
    # title = 'eigenvalue' + str(i+1)
    title = None
    
    err = np.array(errors_[:,i])
    label = str(steps)+'-step'
    
    common = 0#max(err_ext.min(), err.min())
    err_ext = err_ext[err_ext >= common]
    err = err[err >= common]
    
    plot_results(err_ext, label_ext, err, label, title)