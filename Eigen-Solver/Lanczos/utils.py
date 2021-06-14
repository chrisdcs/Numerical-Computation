# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:54:09 2021

@author: m1390
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

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

