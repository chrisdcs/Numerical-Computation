# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:29:15 2021

@author: m1390
"""

import numpy as np


def Arnodi(A,k,b):
    
    Q = np.zeros((A.shape[0],k+1))
    H = np.zeros((k+1,k))
    Q[:,0] = b
    
    for i in range(k):
        v = A @ Q[:,i]
        # orthogonalization
        for j in range(i+1):
            h = Q[:,j] @ v
            # if i < k-1:
            v = v - h * Q[:,j]
            H[j,i] = h
        
        h = np.linalg.norm(v,2)
        H[i+1,i] = h
        Q[:,i+1] = v/h
        
    # solve for first and second eigen values
    # should change to something faster: QR for Hessengurb matrix H
    u,v = np.linalg.eig(H[:-1,:])
    u = abs(u)
    v = abs(v)
    
    idx = np.argsort(u)[::-1]
    
    q = Q[:,:-1] @ v[:,idx[0]]
    
    return q,u[idx[0]],u[idx[1]]