# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:58:14 2015

@author: cdholmes
"""

import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
lowess = sm.nonparametric.lowess

#############################
### Create the pseudo-data

# X coordinate
x = np.sort(np.random.uniform(low = -2*np.pi, high = 2*np.pi, size=1000))
# Y value, truth
ytrue = np.sin(x)
# Y value with noise; Cauchy gives extreme noise
y = ytrue + np.random.normal(size=len(x)) 
y = ytrue + stats.cauchy.rvs(size=len(x)) 


#############################
### Fit the data with lowess

# Lowess, using window width of 1/3
yfit = lowess(y, x, frac=1./3)
# Lowess using a very wide window
z = lowess(y, x)
# Lowess without robust re-weighting of outliers; no outliers are downweighted
u = lowess(y, x, frac=1/3, it=0)

# keep just the fitted values
yfit = yfit[:,1]
# Residuals
yresid = y-yfit

# Find the variance and standard deviation around the fitted line
yvar = lowess(yresid**2, x, frac=1/3)
ystd = np.sqrt(yvar[:,1])


#############################
### Plot the results    

import matplotlib.pyplot as plt
plt.clf()
plt.plot(x,y,'.',label='obs',)
plt.plot(x,ytrue,label='true')
plt.fill_between(x,yfit-ystd,yfit+ystd,alpha=0.5)
plt.plot(x,yfit,'r',label='LOWESS, f=1/3')
plt.plot(x,z[:,1],'g',label='LOWESS, f=1')
plt.plot(x,u[:,1],'c',label='LOWESS, f=1/3, it=0')
plt.ylim([-5,5])
plt.legend()
