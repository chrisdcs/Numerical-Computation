# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:44:22 2021

Gaussian Mixture Model for Generating Dataset

@author: Ding Chi
"""

import numpy as np
from scipy.stats import multivariate_normal as norm

class GMM:
    
    def __init__(self, mu, cov, coeff):
        """
        

        Parameters
        ----------
        mu : ND array
            Mean vectors for the Gaussians.
        cov : ND array
            Covariance matrix for the Gaussians.
        coeff : 1D array
            Mixing coefficients for each Gaussian.

        -------

        """
        
        self.mu = mu
        self.cov = cov
        self.coeff = coeff
        
        self.N = self.mu.shape[0]
        self.D = self.mu.shape[1]
        
        # generate Gaussian distributions
        self.G = [norm(mu[i],cov[i]) for i in range(self.N)]
        
    
    def sample(self,n):
        """
        

        Parameters
        ----------
        n : integer
            Number of samples wish to generate.

        Returns
        -------
        X : ND array
            Data matrix.

        """
        
        X = np.zeros((n,self.D))
        
        interval = np.cumsum(self.coeff)
        
        for i in range(n):
            # generate uniform probability
            U = np.random.rand()
            
            label = np.where(interval > U)[0][0]
            
            X[i] = self.G[label].rvs()
            
        return X