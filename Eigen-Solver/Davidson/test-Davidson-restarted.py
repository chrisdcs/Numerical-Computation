# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:08:10 2021

@author: Chris Ding
"""

import numpy as np
import numpy.linalg as LA
import scipy.sparse as sparse
from utils import Davidson, plot_results, load_sparse_matrix
import time
import argparse
#%% set parameters

parser = argparse.ArgumentParser(description="initialize parameters")

parser.add_argument("--tol", type=float, default=1e-12, help="tolerance for convergence")
parser.add_argument("--data_file_name", type=str, default='data/HBAR_rhf.npz', 
                    help="file directory + file name, e.g. data/TEM27623")
parser.add_argument("--n_eig", type=int, default=10, help="number of eigenvalues to solve")
parser.add_argument("--n_guess", type=int, default=10, help="number of initial guess vectors")
parser.add_argument("--k", type=int, default=10, help="k-step Davidson")
parser.add_argument("--max_iter", type=int, default=100, help="number of max iteration")
parser.add_argument("--descent_order", type=str, default='False', help="solve max/min eigenvalues")
parser.add_argument("--init", type=str, default='random', help="initial guess vector type: 1. random 2. Euclidean")
parser.add_argument("--gamma", type=float, default = -0.1, help="extrapolation parameter  [-1,0)")
parser.add_argument("--compare", type=int, default=1, 
                    help="0: no extrapolation, 1: extrapoltaion, 2: do both and compare")
parser.add_argument("--plot", type=str, default='False', help="whether or not plot residuals")
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
print("compare type:", args.compare)
args.plot = True if args.plot == "True" else False
print("plot type:", args.plot)

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

if args.compare == 1 or args.compare == 2:
    print("\nStart Extrapolated Version!")
    start_davidson_ext = time.time()
    restart, eigenvals, errors = D.restarted_Davidson(V.copy(),True)
    end_davidson_ext = time.time()
    
    # print results after finish
    print("\nExtrapolated Davidson = :", eigenvals[:n_eig],";",
    end_davidson_ext - start_davidson_ext, "seconds")

if args.compare == 0 or args.compare == 2:
    print("\n\nStart Original Version!")
    start_davidson = time.time()
    restart_,eigenvals_,errors_ = D_.restarted_Davidson(V.copy())
    end_davidson = time.time()
    
    # print results after finish
    print("\nDavidson = :", eigenvals_[:n_eig],";",
    end_davidson - start_davidson, "seconds")
#%% Compare with package solvers
# End of Numpy diagonalization. Print results.
start_numpy = time.time()

if sparse.issparse(A):
    A = A.todense()
    
E,Vec = np.linalg.eig(A)
end_numpy = time.time()
idx = E.argsort()
if args.descent_order == 'True':
    idx = idx[::-1]
    E = E[idx]
    print("numpy = ", E[:n_eig], ";",
          end_numpy - start_numpy, "seconds")
else:
    E = E[idx]
    print("numpy = ", E[:n_eig], ";",
          end_numpy - start_numpy, "seconds")
#%% plot results
if args.plot:
    for i in range(n_eig):
        if args.compare == 1 or args.compare == 2:
            err_ext = np.array(errors[:,i])
        else:
            err_ext = None
        label_ext = str(steps)+'-step $γ = {}$'.format(args.gamma)
        title = None
        
        if args.compare == 0 or args.compare == 2:
            err = np.array(errors_[:,i])
        else:
            err = None
        label = str(steps)+'-step'
        
        plot_results(err_ext, label_ext, err, label, title)