# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:17:01 2021

@author: Ding Chi
"""

import h5py
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def plot_results(err_ext,label_ext,err,label,title):
    
    fig,ax = plt.subplots()
    ax.plot(np.log10(err_ext),'-o',label=label_ext)
    ax.plot(np.log10(err),'-^', label=label)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('# of iterations')
    ax.set_ylabel('log residual error')
    plt.show()

class Davidson:
    
    def __init__(self, A, n_eig, n_guess, k, n_iters, tol, restart=True, descent=False):
        
        if n_guess < n_eig: raise Exception('# guess vectors ⩾ # eigen values')
        
        # matrix to solve for eigenvalues and eigenvectors
        self.A = A
        
        
        self.n_eig = n_eig
        self.n_guess = n_guess
        self.n_iters = n_iters
        
        # tolerance for stopping
        self.tol = tol
        
        # m and n are matrix shape values
        self.m = A.shape[0]
        self.n = A.shape[1]
        
        # k step Davidson iteration
        self.k = k
        
        self.done = False
        
        # restart or not
        self.restart = restart
        
        # solving the largest/smallest eigenvalues, default: smallest/descent=False
        self.descent = descent
    
    def iterate(self,V=None):
        
        if V is None:
            # if no initialization then Euclid standard basis
            V = np.zeros((self.m,self.k*self.n_guess))
            V[:,:self.n_guess] = np.eye(self.m,self.n_guess)

        for i in range(1,self.k):
            if (i+1) % 10 == 0:
                print("    Iteration", i+1)
            
            # orthogonalize [V,t]
            Q,_ = np.linalg.qr(V[:,:i*self.n_guess])
            V[:,:i*self.n_guess] = Q
            
            # compute krylov matrix
            H = V[:,:i*self.n_guess].T @ self.A @ V[:,:i*self.n_guess]
            
            # compute ritz vectors and values
            u,v = np.linalg.eig(H)
            if self.descent:
                idx = np.argsort(u)[::-1]
            else:
                idx = np.argsort(u)
            
            u = u[idx]
            v = v[:,idx]
            
            restart_vectors = []
            error = []
            
            # compute residual and approximate next step
            for j in range(self.n_guess):
                restart_vectors.append(V[:,:i*self.n_guess] @ v[:,j])
                
                # compute residuals
                residual = self.A @ restart_vectors[-1]- u[j] * restart_vectors[-1]
                error.append(residual)
                
                # preconditioning
                q = residual/(u[j]-self.A[j,j])
                
                # expand subspace
                V[:,(i*self.n_guess + j)] = q
        
        return np.array(restart_vectors).T, u, np.array(error)
        
    def restarted_Davidson(self,V=None, extrapolate=False):
        
        self.done = False
        errors = []
        
        if V is None:
            previous = np.eye((self.m, self.n_guess))
        else:
            previous = V[:,:self.n_guess]
        
        for i in range(self.n_iters):
            
            restart, eigenvals, error = self.iterate(V)
            val = np.linalg.norm(error,2,axis=1)
            
            if (val[:self.n_eig] < self.tol).all():
                self.done = True
                
            errors.append(val)
            print("Epoch",i,"error:",np.linalg.norm(val[:self.n_eig]))
            
            if self.done:
                return restart, eigenvals, np.array(errors)
            
            if extrapolate:
                # gamma = -(eigenvals[:self.n_guess]/eigenvals[1:self.n_guess+1]) ** (i+1)
                # gamma = np.array(self.n_guess * [-0.75])
                gamma = np.array(self.n_guess * [-0.5])
                gamma.reshape(1,-1)
                V = np.zeros((self.m,self.k * self.n_guess))
                V[:,:self.n_guess] = (1-gamma) * restart + gamma * previous
                previous = restart.copy()
            else:
                V = np.zeros((self.m,self.k * self.n_guess))
                V[:,:self.n_guess] = restart
        print("Done!")
            
        return restart, eigenvals, np.array(errors)
    
    
def load_sparse_matrix(filename):
    
    # load sparse matrix from .mat file in read mode
    f = h5py.File(filename, 'r')
    A = f['Problem']['A']
    
    ir = np.array(A['ir'])
    jc = np.array(A['jc'])
    
    data = np.array(A['data'])
    row_idx = np.array(ir)
    col_idx = []
    
    count = 0
    for idx in range(data.shape[0]):
        if jc[count] == idx:
            col_idx.append(count)
            count += 1
        else:
            col_idx.append(col_idx[-1])
    
    col_idx = np.array(col_idx).astype(np.uint64)
    
    mat = csr_matrix((data,(row_idx,col_idx)))
    
    return mat
