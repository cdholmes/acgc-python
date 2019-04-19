# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 13:36:57 2015

@author: cdholmes
"""

#conda install --channel https://conda.anaconda.org/r rpy2

import rpy2.robjects as robjects
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t 

#from rpy2.robjects.packages import importr
#msir = importr("msir")

# Evaluate a built-in constant 
pi = robjects.r('pi')[0]
print(pi)

# Define a vector
res = robjects.FloatVector([1.1, 2.2, 3.3])
print(res.r_repr())

# Define a matrix
v = robjects.FloatVector([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
m = robjects.r['matrix'](v, nrow = 2)
print(m)

# Call a function
rsum = robjects.r['sum']
print(rsum(robjects.IntVector([1,2,3]))[0])

# Call a function with keywords
rsort = robjects.r['sort']
res = rsort(robjects.IntVector([1,2,3]), decreasing=True)
print(res.r_repr())

# Call loess
N = 1000
x = np.arange(N)
y = np.cos(x*2*np.pi/365) + 0.3*np.random.standard_cauchy(N)#0.3*np.random.randn(N)#0.3*(np.abs(np.sin(x*2*np.pi/500)))*np.random.randn(N)#  
from loess import loess
yfit, ystd, yste = loess(x,y,span=0.3)

plt.clf()
plt.plot(x,y,'.')
plt.fill_between(x,yfit-ystd,yfit+ystd,alpha=0.5)
plt.plot(x,yfit,'-')

plt.ylim([-5,5])

