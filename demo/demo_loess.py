
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:40:56 2015

@author: cdholmes
"""
import numpy as np
from acgc.stats import loess
import matplotlib.pyplot as plt

# Create a synthetic dataset -- sinusoid with noise
N = 1000
x = np.arange(N)
y = np.cos(x*2*np.pi/365) + 0.3*np.random.standard_cauchy(N)
y = np.cos(x*2*np.pi/365) + 0.5*np.random.randn(N)
# add some outliers
y[np.array([10,100,500,750])] = [4,4.5,-10,-4.5]
# add some NaNs
y[200]=np.nan

# fit with loess
yfit, ystd, yste = loess(x,y,span=0.3)

plt.clf()
plt.plot(x,y,'k.',label='data')
plt.fill_between(x,yfit-2*ystd,yfit+2*ystd,alpha=0.5,label='2 sigma')
plt.plot(x,yfit,'-',label='loess fit')
plt.ylim([-5,5])
plt.legend(loc='best')
