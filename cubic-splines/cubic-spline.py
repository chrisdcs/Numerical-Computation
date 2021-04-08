# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:56:28 2021

@author: m1390
"""

import numpy as np
import matplotlib.pyplot as plt
#%% (a)
def tridiag(n,u,h,v):
    """
    This function solves a tridiagonal system specifically for
    natural cubic splines. (reduced)
    
    More details see hw6.pdf problem # 5(a).

    Parameters
    ----------
    n : int
        n+1 points
    u : n-1 dim vector 1,...,n-1
        diagonal elements.
    h : n-1 dim vector
        tridiagonal elements.
    v : n-1 dim vector
        objective for system.

    Returns
    -------
    z : n-1 dim vector
        system coefficients.

    """
    
    # Gaussian elimination step
    for k in range(1,n-1):
        u[k] = u[k] - h[k-1]**2/u[k-1]
        v[k] = v[k] - h[k-1] * v[k-1]/u[k-1]
    
    # back substitution step
    z = np.zeros((n-1))
    z[-1] = v[-1]/u[-1]
    for k in range(2,n):
        z[-k] = (v[-k] - h[-k] * z[-(k-1)]) / u[-k]
    
    return u,v,z
#%% (b) Natural cubic splines
# Implementation based on notes from hw6.pdf problem #5

def natural_spline(x,f,plot=False):
    """
    

    Parameters
    ----------
    x : vector [x0,x1,x2,...,xn]
        distinct points.
    f : numpy function
    
    plot : Boolean, optional
        decide plot results or not. The default is False.

    Returns
    -------
    errmax : list
        list of max errors.

    """
    
    # x,y are n+1 dim vector
    n = x.shape[0] - 1
    
    # y : vector [y0,y1,y2,...,yn] function evaluations on x.
    y = f(x)
    
    # obtain tridiagonal system (details please see hw6.pdf problem # 5(b))
    # subinterval lengths h: n dim vector (from 0 to n-1)
    h = x[1:] - x[:-1]
    v = 6 * (1/h[1:] * (y[2:]-y[1:-1]) - 1/h[:-1]*(y[1:-1] - y[:-2]))
    u = 2 * (h[:-1]+h[1:])
    
    _,_,z = tridiag(n, u, h[1:], v)
    
    z = np.concatenate(([0], z, [0]))
    # compute coefficients for natural splines
    coefficients = [np.array([zi1/(6*hi),
                              zi/(6*hi),
                              (yi1/hi-hi/6*zi1),
                              yi/hi-hi/6*zi])
                    for (zi,zi1,hi,yi,yi1) in zip(z[:-1],z[1:],h,y[:-1],y[1:])]
    
    # evaluate natural cubic spline
    N = 51
    errmax = []
    
    for j in range(n):
        # loop over sub-intervals
        coeff = coefficients[j]
        
        # 2 boundary points for interval j
        bdry = x[j:j+2]
        
        # equally sample 51 points from interval j
        Ij = np.linspace(bdry[0],bdry[1],N).reshape(-1,1)
        
        # obtain data matrix 
        X = np.concatenate([
                (Ij - bdry[0])**3,
                (bdry[1] - Ij)**3,
                (Ij - bdry[0]),
                (bdry[1] - Ij)
            ],1)
        
        # reduce back to original shape
        Ij = Ij.flatten()
        
        # evaluate on piecewise cubic spline
        Sj = X @ coeff
        
        errmax.append(np.max(
                                np.abs(
                                        Sj - f(Ij)
                                        )
                            )
                        )
        
        if j == 0:
            S = Sj.copy()
            function = f(Ij)
            I = Ij.copy().flatten()
        else:
            S = np.concatenate([S,Sj])
            function = np.concatenate([function,f(Ij)])
            I = np.concatenate([I,Ij.flatten()])
            
    if plot:
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)  
          
        # Ensure that the axis ticks only show up on the bottom and left of the plot.  
        # Ticks on the right and top of the plot are generally unnecessary chartjunk.  
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
        
        ax.scatter(x,y,color='chocolate',label='nodes')
        ax.plot(I,S,label='NS')
        ax.plot(I,function,color='grey',label='f')
        
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')

    return errmax
#%% (d)
"""
    Comment and uncomment pieces of code to test different cases
"""

def x52(x):
    return x**(5/2)

def exp_(x):
    return np.exp(-x)

# This is a visualization demo
##############################################################################
x = np.linspace(0, 6*np.pi, 13)
f = np.sin
error = natural_spline(x, f, True)
##############################################################################