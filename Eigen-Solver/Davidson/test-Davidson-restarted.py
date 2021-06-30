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
<<<<<<< HEAD

tol = 1e-8			        # Convergence tolerance
data_file_name = r'data/rhfHBAR.npz'
n_eig = 5                  # number of eigen values to compute
n_guess = 10                # number of initial guess vectors: could be larger than 1 for each eigenvalue
k = 4                       # k-step Davidson
steps = k                   # number of steps
max_iter = 300              # max number of times to run restarted Davidson
descent_order = False       # if descent order True, we are solving max eigenvalues, vice versa
init = 'Euclidean'          # type of guess vector initialization

=======
parser = argparse.ArgumentParser(description="initialize parameters")

parser.add_argument("--tol", type=float, default=1e-14, help="tolerance for convergence")
parser.add_argument("--data_file_name", type=str, default='data/rhfHBAR.npz', 
                    help="file directory + file name, e.g. data/TEM27623")
parser.add_argument("--n_eig", type=int, default=5, help="number of eigenvalues to solve")
parser.add_argument("--n_guess", type=int, default=5, help="number of initial guess vectors")
parser.add_argument("--k", type=int, default=20, help="k-step Davidson")
parser.add_argument("--max_iter", type=int, default=500, help="number of max iteration")
parser.add_argument("--descent_order", type=str, default='False', help="solve max/min eigenvalues")
parser.add_argument("--init", type=str, default='random', help="initial guess vector type: 1. random 2. Euclidean")
parser.add_argument("--gamma", type=float, default = -0.1, help="extrapolation parameter  [-1,0)")

args = parser.parse_args()

print()
print("tolerance:", args.tol,)
print("data file name:", args.data_file_name)
print("number of eigenvalues to solve:", args.n_eig)
print("number of initial guess vectors (block size):", args.n_guess)
print("{}-step Davidson".format(args.k))
print("max number of iterations:", args.max_iter)
if args.descent_order=="True":
    print("solve max eigenvalues")
elif args.descent_order == "False":
    print("solve min eigenvalues")
else:
    raise Exception("Invalid Bool Input")
print("initial guess vectors:", args.init)
print("extrapolation parameter gamma:", args.gamma)

tol = args.tol                          # Convergence tolerance
data_file_name = args.data_file_name    #'data/TEM27623.mat'
n_eig = args.n_eig                      # number of eigen values to compute
n_guess = args.n_guess                  # number of initial guess vectors: could be larger than 1 for each eigenvalue
k = args.k                              # k-step Davidson
steps = k                               # number of steps
max_iter = args.max_iter                # max number of times to run restarted Davidson
descent_order = True if args.descent_order == "True" else False       # if descent True, we are solving max eigenvalues, vice versa
init = args.init                        # type of guess vector initialization
gamma = args.gamma
>>>>>>> argparse
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
    
# eigvec = np.load('eigvec.npy')
# V[:,:5] = eigvec[:,:5]
#%% Initialize algorithm and computation

D = Davidson(A, n_eig, n_guess, steps, max_iter, tol, gamma, descent = descent_order)
D_ = Davidson(A, n_eig, n_guess, steps, max_iter, tol, descent = descent_order)

print("\nStart Extrapolated Version!")
start_davidson = time.time()


"""
    Initialization matters, if we use random vectors, converges very slowly
"""
restart, eigenvals, errors = D.restarted_Davidson(V.copy(),True)
end_davidson = time.time()

print("\n\nStart Original Version!")
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

#%% plot results
for i in range(n_eig):
    err_ext = np.array(errors[:,i])
    label_ext = str(steps)+'-step $γ = {}$'.format(args.gamma)
    # title = 'eigenvalue' + str(i+1)
    title = None
    
    err = np.array(errors_[:,i])
    label = str(steps)+'-step'
    
    common = 0#max(err_ext.min(), err.min())
    err_ext = err_ext[err_ext >= common]
    err = err[err >= common]
    
    plot_results(err_ext, label_ext, err, label, title)

# np.save('eigvec.npy',restart)
# np.save('eigval.npy',eigenvals)

