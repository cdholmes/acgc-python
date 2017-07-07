#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 00:17:00 2017

@author: cdholmes
"""

import numpy as np
from sma_robust_fit import robust_cov
import os
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Versions/Current/Resources/bin/'
import scipy.linalg as la

x= np.array([1,2,3,4])
y= np.array([2,4,6,10])

N = 1000
x = np.random.randn(N)
y = np.random.randn(N)

#robust_cov(x,y)

N = len(x)
    
X = np.array([x,y])
    
# first guess is normal covariance
rcov = np.cov( X )
    
# first guess is normal means
rm = X.mean(axis=1)

c=3
#c=1.345

it = 0
d1 = 1
d2 = 1
eps = 1e-6 
while ((it<10) and (d1>eps) and (d2>eps) ):    

    # inverse square root of covariance matrix
    rinvsq = la.sqrtm( la.inv( rcov ) ) 
    
    # z-scores
    z = la.norm( rinvsq @ ( X - np.tile(rm,[N,1]).T ) , axis=0 )

    # Weights for mean
    w1 = np.minimum( 1, c/z )
    #w1 = np.ones(N)

    # Weights for covariance, add scaling later
    w2 = w1**2 

    # New estimate of robust mean
    rm2 = np.sum( X * np.tile(w1,[2,1]), axis=1 ) / np.sum(w1)
    
    # Means in matrix form
    Xm = np.tile(rm2,[N,1]).T

    # New estimate of robust covariance
    rcov2 = 1/(N-1) * ( (X-Xm) @ ( (X-Xm) * np.tile(w2,[2,1]) ).T )
    
    it += 1
    d1 = np.max(np.abs(rm2-rm))
    d2 = np.max(np.abs(rcov2-rcov))
    rcov = rcov2
    rm   = rm2
    
print(it, rm)