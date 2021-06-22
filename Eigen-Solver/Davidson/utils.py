# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:17:01 2021

@author: Ding Chi
"""

import h5py
import numpy as np
import scipy.sparse as sparse
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
        self.eigVec = {i: None for i in range(n_guess)}
        self.residuals = {i: None for i in range(n_guess)}
        
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

        # keep track of subspace size
        size = self.n_guess
        
        for i in range(1,self.k):
            # iterate k-step Davidson
            
            if (i+1) % 10 == 0:
                print("    Step", i+1)
            
            # orthogonalize [V,t]
            Q,_ = np.linalg.qr(V[:,:size])
            V[:,:size] = Q
            
            # compute krylov matrix
            H = V[:,:size].T @ self.A @ V[:,:size]
            
            # compute eigen vectors and values for H
            u,v = np.linalg.eig(H)
            if self.descent:
                idx = np.argsort(u)[::-1]
            else:
                idx = np.argsort(u)
            
            u = u[idx]
            v = v[:,idx]
            
            restart_vectors = np.zeros((self.m,self.n_guess))
            residuals = np.zeros((self.m,self.n_guess))
            
            # increament subspace V, but not use them until it is orthogonalized
            curr_size = size
            # compute residual and approximate next step
            for j in range(self.n_guess):
                
                if self.eigVec[j] is not None:
                    restart_vectors[:,j] = self.eigVec[j]
                    residuals[:,j] = self.residuals[j]
                    continue
                
                # compute ritz vectors
                ritz_vector = V[:,:curr_size] @ v[:,j]
                restart_vectors[:,j] = ritz_vector
                
                # compute residuals
                residual = self.A @ restart_vectors[:,j]- u[j] * restart_vectors[:,j]
                if np.linalg.norm(residual) < self.tol:
                    self.eigVec[j] = ritz_vector
                    self.residuals[j] = residual
                
                residuals[:,j] = residual
                
                # preconditioning
                q = residual/(u[j]-self.A[j,j])
                
                # expand subspace
                V[:,size] = q
                # increament subspace size
                size += 1
        
        return restart_vectors, u, residuals.T
        
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
                print("Done!")
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
            
        return restart, eigenvals, np.array(errors)
    
    
def load_sparse_matrix(filename):
    
    _,ext = filename.split('.')
    
    if ext == 'mat':
    
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
    
    elif ext == 'npz':
        mat = sparse.load_npz(filename)
    elif ext == 'npy':
        mat = np.load(filename)
    else:
        raise Exception("File type not supported!");
    
    return mat
