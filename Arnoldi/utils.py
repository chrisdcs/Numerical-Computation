# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:29:15 2021

@author: m1390
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

class Arnoldi:
    
    def __init__(self,A,k,tol):
        """
        

        Parameters
        ----------
        A : N-D matrix
            The matrix to solve for eigenvalues and eigenvectors.
        k : int
            k step Arnoldi iteration.
        tol : double
              The tolerance for stopping the algorithm

        Returns
        -------
        None.

        """
        self.A = A
        self.k = k
        # error list for storing eigenvector residuals
        self.err_eigvector = []
        self.err_eigval = []
        
        self.tol = tol
        
    def reset(self,A,k):
        if A is not None:
            self.A = A
        if k:
            self.k = k
        self.err_eigval = []
        self.err_eigvector = []
        
    def iterate(self, b, A = None, k = None):
        # This is the k-step Arnoldi iteration
        
        if A is None:
            A = self.A.copy()
        if k == None:
            k = self.k
        
        # Krylov span vectors
        Q = np.zeros((A.shape[0],k+1))
        
        # upper Hessenburg matrix H
        H = np.zeros((k+1,k))
        
        if np.linalg.norm(b) != 1:
            b /= np.linalg.norm(b)
            
        Q[:,0] = b
        
        for i in range(k):
            v = A @ Q[:,i]
            # orthogonalization
            for j in range(i+1):
                h = Q[:,j] @ v
                v = v - h * Q[:,j]
                H[j,i] = h
            
            h = np.linalg.norm(v,2)
            H[i+1,i] = h
            Q[:,i+1] = v/h
            
        # solve for first and second eigen values
        ####### should change to something faster #######
        u,v = np.linalg.eig(H[:-1,:])
        u = np.real(u)
        v = np.real(v)
        
        # sort things in descending order
        idx = np.argsort(u)[::-1]
        
        q = Q[:,:-1] @ v[:,idx[0]]
        
        return q,u[idx[0]],u[idx[1]]
    
    def restarted_arnoldi(self, y, extrapolate, n_iter, 
                          A = None, k = None, true_eigenval = None):
        """
        

        Parameters
        ----------
        y : vector
            the vector that forms the Krylov space.
        extrapolate : Boolean
            Do extrapolation or not.
        n_iter : TYPE
            Number of iterations for restarted Arnoldi method.
        A : TYPE, optional
            The matrix to solve for eigenvalues. The default is None.
        k : TYPE, optional
            k step Aronoldi iteration. The default is None.
        true_eigenval : double, optional
            True eigenvalue that we are solving. The default is None.

        Returns
        -------
        None.

        """
        try:
            extrapolate == True or extrapolate == False
        except:
            print("extrapolate variable must be Boolean type!")
        
        self.reset(A, k)
        
        A = self.A.copy()
        k = self.k
        
        u = y
        for i in range(n_iter):
            # k step Arnoldi iteration
            y_, lam0, lam1 = self.iterate(u, A)
            
            # extrapolate
            if extrapolate:
                gamma = -abs(lam1/lam0)**i
                # gamma = -0.75
            else:
                gamma = 0
            u = (1-gamma) * y_ + gamma * y
            y = y_
            
            # record errors
            err = np.linalg.norm(A @ y_ - lam0 * y_,2)
            self.err_eigvector.append(err)
            if true_eigenval:
                self.err_eigval.append(abs(lam0-true_eigenval))
            
            # stop criteria
            if err < self.tol:
                break
    
def plot_results(A_ext,label_ext,Arn,label,title):
    
    fig,ax = plt.subplots()
    ax.plot(np.log10(A_ext.err_eigvector),'-o',label=label_ext)
    ax.plot(np.log10(Arn.err_eigvector),'-^', label=label)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('# of iterations')
    ax.set_ylabel('log residual error')
    
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