# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:08:24 2021

@author: Chi Ding

Conjugate Gradient:
    Ax = b
    
    A: Hilbert matrix
    x: coefficients
    b: ones vector
"""

import numpy as np

# dimensionality for the system
D = 5,8,12,20

# stopping criterion
threshold = 1e-6

A = []

x = []


for n in D:
    
    k = 0
    
    A.append(
                np.column_stack([1/(i+j-1) 
                         for i in range(1,n+1) 
                         for j in range(1,n+1)]).reshape(n,n)
                )
    
    b = np.ones(n)
    
    # initial point
    x.append(np.zeros(n))
    
    # initial residual
    r = A[-1] @ x[-1] - b
    
    p = -r
    
    while True:
        """
            The conjugate gradient algorithm
        """
        r2 = r @ r
        alpha = r2 / (p @ A[-1] @ p)
        x[-1] = x[-1] + alpha*p
        r_ = r + alpha * A[-1] @ p
        beta = (r_ @ r_)/r2
        p = -r_ + beta * p
        r = r_.copy()
        k += 1
        
        if (r @ r)**0.5 < threshold:
            break
        
    print('dimension',n,'steps',k)