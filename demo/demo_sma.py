# -*- coding: utf-8 -*-
"""Demonstrate standard major axis (SMA) fitting

@author: cdholmes
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from acgc.stats import sma
import statsmodels.formula.api as smf

#%%
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
result = sma(x,y,cl=0.95,robust=True)
# SMA fit on the "pure" data without outliers
#sp,ip,stdsp,stdip,cisp,ciip = smafit(x,ypure,cl=0.95)

# OLS for comparison
res = smf.ols('y ~ x + 1',{'x':x, 'y':y}).fit()

fmt = '{:15s}{:8.4f}+/-{:8.4f}'

print(fmt.format('slope',result['slope'],result['slope_ste']))
print(fmt.format('intercept',result['intercept'],result['intercept_ste']))

plt.clf()
#plt.style.use('bmh')
plt.scatter(x,y,label='data') 
plt.plot(x,x,label='1:1')
plt.plot(x,result['slope']*x+result['intercept'],label='SMA')
#plt.plot(x,sp*x+ip,label='SMA (pure)')
plt.plot(x,res.fittedvalues,label='OLS')
plt.legend()
plt.show()


