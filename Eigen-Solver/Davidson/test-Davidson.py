# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:07:00 2021

@author: Chris Ding
"""

import numpy as np
import time

n = 1200					# Dimension of matrix
tol = 1e-7				# Convergence tolerance

sparsity = 0.01
A = np.zeros((n,n))

for i in range(0,n):
    A[i,i] = i + 1 
A = A + sparsity*np.random.randn(n,n) 
A = (A.T + A)/2 


I = np.eye(n)
# k step Lanczos iteration
l = 5                  # number of initial guess vectors
eig = 5                # number of eigen values to compute
steps = 100            # number of steps: k

# number of initial guess must be larger or equal to number of eigen values
if l < eig: raise Exception('l must be >= eig')

start_davidson = time.time()


V = np.zeros((n,steps*l))
# initialize guess vectors
"""
    Initialization matters, if we use random vectors, converges slower, but doesn't
    always work.
"""
# for i in range(l):
#     v0 = np.random.rand(n)
#     V[:,i] = v0/np.linalg.norm(v0)

V[:,:l] = np.eye(n,l)

u_old = 1

for i in range(1,steps):
    
    Q,_ = np.linalg.qr(V[:,:i*l])
    V[:,:i*l] = Q
    
    H = V[:,:i*l].T @ A @ V[:,:i*l]
    
    u,v = np.linalg.eig(H)
    idx = np.argsort(u)
    u = u[idx]
    v = v[:,idx]
    
    if np.linalg.norm(u[:eig]-u_old) < tol:
        break
    u_old = u[:eig]
    
    for j in range(l):
        w = (A - u[j]*I) @ (V[:,:i*l] @ v[:,j])
        q = w/(u[j]-A[j,j])
        V[:,(i*l+j)] = q
        
end_davidson = time.time()
    
print("davidson = ", u[:eig],";",
    end_davidson - start_davidson, "seconds")


start_numpy = time.time()

E,Vec = np.linalg.eig(A)
E = np.sort(E)

end_numpy = time.time()

# End of Numpy diagonalization. Print results.

print("numpy = ", E[:eig],";",
     end_numpy - start_numpy, "seconds") 