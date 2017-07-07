# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:37:46 2016

@author: cdholmes
"""

import numpy as np
import matplotlib.pyplot as plt
from sma_fit import sma_fit
import statsmodels.formula.api as smf

# Create 100 normally distributed points with slope 2, intercept 1
n = 1000
x0 = np.random.randn(n)
x = x0 + np.random.randn(n)
y = x0*2 + 1 + np.random.randn(n)

# Add some missing elements
#x[1] = np.nan
#y[10] = np.nan

# Add some outliers
idxout = [750,800,850,950]
ypure = np.copy(y)
#ypure[idxout] = np.nan
y[idxout] = [400,300,450,420]

# SMA fit, use robust methods to minimize effect of outliers
s,i,stds,stdi,cis,cii = sma_fit(x,y,cl=0.95,robust=True)
# SMA fit on the "pure" data without outliers
#sp,ip,stdsp,stdip,cisp,ciip = sma_fit(x,ypure,cl=0.95)

# OLS for comparison
res = smf.ols('y ~ x + 1',{'x':x, 'y':y}).fit()

fmt = '{:15s}{:8.4f}+/-{:8.4f}'

print(fmt.format('slope',s,stds))
print(fmt.format('intercept',i,stdi))

plt.clf()
#plt.style.use('bmh')
plt.scatter(x,y,label='data') 
plt.plot(x,x,label='1:1')
plt.plot(x,s*x+i,label='SMA')
#plt.plot(x,sp*x+ip,label='SMA (pure)')
plt.plot(x,res.fittedvalues,label='OLS')
plt.legend()
plt.show()


