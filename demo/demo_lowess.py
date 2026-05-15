# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:58:14 2015

@author: cdholmes
"""

import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
lowess = sm.nonparametric.lowess

#############################
### Create the pseudo-data

# X coordinate
x = np.sort(np.random.uniform(low = -2*np.pi, high = 2*np.pi, size=1000))
# Y value, truth
ytrue = np.sin(x)
# Y value with noise; Cauchy gives extreme noise
y = ytrue + 3*np.random.normal(size=len(x)) 
# y = ytrue + stats.cauchy.rvs(size=len(x)) 


#############################
### Fit the data with lowess

# Adjust frac between 0 and 1 for your specific data
# frac is the fraction of the data used for estimating each yfit value
# large frac gives a smoother fit, but may miss local features; 
# small frac gives a more wiggly fit, but may be more sensitive to noise
# Lowess, using window width of 1/3
yfit = lowess(y, x, frac=1./3, return_sorted=False )

# Residuals
yresid = y-yfit

# Find the variance and standard deviation around the fitted line
yvar = lowess(yresid**2, x, frac=1/3, return_sorted=False)
ystd = np.sqrt(yvar)

#############################
### Try other parameter values to illustrate their effect

# Lowess using a very wide window
z = lowess(y, x, frac=1, return_sorted=False)
# Lowess without robust re-weighting of outliers; no outliers are downweighted
u = lowess(y, x, frac=1/3, it=0, return_sorted=False)

#############################
### For periodic or cyclic data, 
# we can extend the data by wrapping it around the ends.

# The period of the data; 
# here we know it is 4*pi (-2*pi to 2*pi),
# but it will vary for different data
period = 4*np.pi 

# Repeat the data at the ends, shifted by the period
xext = np.concatenate((x-period,x,x+period))
yext = np.concatenate((y,y,y))

# Fit the extended data with lowess
# Note that frac should be 3x smaller than for the original data, 
# since we have 3x as many points
yfit_periodic = lowess(yext, xext, frac=1./9, return_sorted=False )
# yfit_periodic = yfit_periodic[:,1]

# Trim the fitted values to the original x range
# This isn't strictly necessary, but it makes it easier to compare with the original data
yfit_periodic = yfit_periodic[len(x):2*len(x)]

#############################
### Plot the results    

plt.clf()
plt.plot(x,y,'.',label='obs',color='gray')
plt.plot(x,ytrue,'k',label='true')
plt.fill_between(x,yfit-ystd,yfit+ystd,alpha=0.5,color='r')
plt.plot(x,yfit,'r',label='LOWESS, f=1/3')
plt.plot(x,yfit_periodic,'m',label='LOWESS, f=1/3, periodic')
plt.plot(x,z,'g',label='LOWESS, f=1')
plt.plot(x,u,'c',label='LOWESS, f=1/3, it=0')

plt.ylim([-5,5])
plt.legend()
