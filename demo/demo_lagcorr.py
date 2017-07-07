# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:47:41 2015

@author: cdholmes
"""

import numpy as np
import matplotlib.pyplot as p
import scipy.stats as stats
from boxcar import boxcar
from tapply import tapply
from boxcarpoly import boxcarpoly

# Create a AR(1) time series
N = 365
a = np.random.randn(N)
for i in range(1,N):
    a[i] = a[i-1]*0.8 + a[i]*0.2

# Create 2 correlated time series with annual and semi-annual cycles
t = np.arange(365)
seasonal = 1-np.cos(2*np.pi*t/365) + 0.2*(1-np.cos(4*np.pi*t/365))
noise = 0.3*np.random.randn(N)
b = seasonal + a
c = seasonal + a + noise

b[100]=np.nan
b[200]=20
# remove the seasonal cycle from b and c
width = 40
bds = b-boxcarpoly(b,width,order=2)
cds = c-boxcarpoly(c,width,order=2)

NL = 90
corr = np.zeros(NL)
corrp = np.zeros(NL)
corrt = np.zeros(NL)
for i in range(NL):
    # lag correlation between deseasonalized a and b
    result = stats.spearmanr(bds[i:-1],cds[0:-(i+1)])
    corr[i] = result[0]
    corrp[i] = result[1]
    # true correlation between a and b, if seasonal cycle were removed perfectly
    corrt[i] = stats.spearmanr(a[i:-1],(a+noise)[0:-(i+1)])[0]
    
p.clf()
p.subplot(3,1,1)
p.plot(b)
p.plot(c)
p.plot(boxcar(b,width))
p.plot(boxcarpoly(b,width,order=2))
p.subplot(3,1,2)
p.plot(bds)
p.plot(cds)
p.subplot(3,1,3)
p.plot(corr)
p.plot(corrt)
p.plot(corrp>0.05)
