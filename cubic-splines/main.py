# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:36:16 2021

@author: m1390
"""


import numpy as np
import sklearn.datasets as D
from CubicSpline import Cubic_Spline

# a simple sine wave demo
x = np.linspace(0, 6*np.pi, 13)
f = np.sin
y = f(x)

NCS = Cubic_Spline(x, y, f)
NCS.train()
NCS.evaluate(20)
NCS.plot()

# swiss roll demo
n = 100
data, _ = D.make_swiss_roll(n)
data =np.vstack([data[:,0],data[:,2]]).T
norm = np.linalg.norm(data,2,axis=1)
data = data[np.argsort(norm)]

x = data[:,0]
y = data[:,1]

NCS.reset(x, y)
NCS.train()
NCS.evaluate(30)
NCS.plot()