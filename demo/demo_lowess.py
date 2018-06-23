# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:58:14 2015

@author: cdholmes
"""

import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

lowess = sm.nonparametric.lowess
x = np.sort(np.random.uniform(low = -2*np.pi, high = 2*np.pi, size=1000))
ytrue = np.sin(x)
y = ytrue + stats.cauchy.rvs(size=len(x))#np.random.normal(size=len(x))
yfit = lowess(y, x, frac=1./3)
z = lowess(y, x)
u = lowess(y, x, frac=1/3, it=0)

# keep just the fitted values
yfit = yfit[:,1]
# Residuals
yresid = y-yfit

# Find the variance and standard deviation around the fitted line
yvar = lowess(yresid**2,x,frac=1/3)
ystd = np.sqrt(yvar[:,1])


    
import matplotlib.pyplot as p
p.clf()
p.plot(x,y,'.',label='obs')
p.plot(x,ytrue,label='true')
p.fill_between(x,yfit-ystd,yfit+ystd,alpha=0.5)
p.plot(x,yfit,'r',label='LOWESS, f=1/3')
p.plot(x,z[:,1],'g',label='LOWESS, f=1')
p.plot(x,u[:,1],'c',label='LOWESS, f=1/3, it=0')
p.ylim([-5,5])
p.legend()