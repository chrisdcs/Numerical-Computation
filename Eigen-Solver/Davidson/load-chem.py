# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:51:54 2021

@author: m1390
"""

from functools import partial
from scipy.sparse import csr_matrix

n = 10100

with open(r'F:/Download/rhfHBAR.txt', 'r+b') as f:
    data = []
    row_idx = []
    col_idx = []
    count = 0
    f_read  = partial(f.readline)
    for text in iter(f_read,''):
        txt = text.decode()
        # detect end of file
        if txt != '':
            num = float(text.decode())
            if abs(num) > 0:
                data.append(num)
                row_idx.append(count % n)
                col_idx.append(count // n)
            count += 1
        else:
            break
        
mat = csr_matrix((data,(row_idx, col_idx)))