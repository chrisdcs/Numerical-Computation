# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:56:28 2021

@author: m1390
"""

import numpy as np
import matplotlib.pyplot as plt
#%%
class Cubic_Spline:
    
    def __init__(self, x, y, f=None):
        """
        

        Parameters
        ----------
        x : data points
            N-D vector.
        y : data evaluations
            N-1 vector.
        f: explicit function
            Function

        Returns
        -------
        None.

        """
        self.x = x
        self.y = y
        self.f = f
        
        # number of subintervals
        self.n = self.x.shape[0] - 1
        
        # piece-wise coefficients of the cubic splines
        self.coefficients = None
        
        # I: interval for evaluation
        # S: evaluation for interval
        self.I = None
        self.S = None
        
    def reset(self,x,y,f=None):
        self.__init__(x,y,f)
        
    def tridiag(self,u,h,v):
        """
        This function solves a tridiagonal system specifically for
        natural cubic splines. (reduced)
        
        It can be generalized for other cubic splines.
    
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
        n = self.n
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
    
    def train(self):
        
        # data points x
        x = self.x
        
        # y : vector [y0,y1,y2,...,yn] function evaluations on x.
        y = self.y
        
        # obtain tridiagonal system
        # subinterval lengths h: n dim vector (from 0 to n-1)
        h = x[1:] - x[:-1]
        v = 6 * (1/h[1:] * (y[2:]-y[1:-1]) - 1/h[:-1]*(y[1:-1] - y[:-2]))
        u = 2 * (h[:-1]+h[1:])
        
        _,_,z = self.tridiag(u, h[1:], v)
        
        z = np.concatenate(([0], z, [0]))
        
        # compute coefficients for natural splines
        self.coefficients = [np.array([zi1/(6*hi),
                                       zi/(6*hi),
                                       (yi1/hi-hi/6*zi1),
                                       yi/hi-hi/6*zi])
                             for (zi,zi1,hi,yi,yi1) in 
                             zip(z[:-1],z[1:],h,y[:-1],y[1:])]
    
    def evaluate(self,N):
        # evaluate natural cubic spline
        # N is number of points to sample for each subinterval
        
        n = self.n
        
        x = self.x
        
        for j in range(n):
            
            # loop over sub-intervals
            coeff = self.coefficients[j]
            
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
            
            
            if j == 0:
                S = Sj.copy()
                I = Ij.copy().flatten()
            else:
                S = np.concatenate([S,Sj])
                I = np.concatenate([I,Ij.flatten()])
        
        self.S = S
        self.I = I
        
    def plot(self):
        plt.figure()
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)  
          
        # Ensure that the axis ticks only show up on the bottom and left of the plot.  
        # Ticks on the right and top of the plot are generally unnecessary chartjunk.  
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
            
        ax.scatter(self.x,self.y,color='chocolate',label='nodes')
        ax.plot(self.I,self.S,label='NS')
        
        if self.f is not None:
            function = self.f(self.I)
            ax.plot(self.I,function,color='grey',label='f')
        
        
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
