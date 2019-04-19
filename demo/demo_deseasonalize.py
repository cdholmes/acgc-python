# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 18:33:31 2015

@author: cdholmes
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn



def ds(a,t=[0]):
    # deseasonalize
    # --remove a seasonal cycle by subtracting the monthly 
    #   deviations from the long-term mean
    # a = 1-D array of data to deseasonalize
    # t = 1-D array of time coordinates corresponding to array a (optional)
    #     t can be pandas DatetimeIndex OR
    #     t can be in units of months since beginning of time series (typically an integer)
     
    from tapply import tapply
    import numpy as np

    if (len(t) != 1 and len(t) != len(a)):
        print('Array t must have same length as a')
    
    if( len(t)==1 ):
        g = np.arange(a.size) % 12
    else:
        try:
            g=[]
            for x in t:
                g.append(float(x.month))
            g=np.array(g)
        except: 
            g = np.floor( t % 12 )

    m0 = np.nanmean( a )
    
    m, gm = tapply( a - m0, g, np.nanmean )
    b = np.zeros( a.size )
    for i, gmi in enumerate(gm):
        ind = np.where( g == gmi )
        b[ind] = a[ind] - m[i]
    
    return b

t=np.arange(96)+10
tt = pd.date_range(start='1/1/2001',periods=96,freq='MS')
a = np.sin(t*2*np.pi/12) + t/48 + 0.2*np.random.randn(t.size) +5
a[10:20] =np.nan
a_ds = ds(a,t)
a_ds2 = ds(a,tt)
plt.clf()
plt.subplot(2,1,1)
plt.plot(t,a,label='original')
plt.plot(t,a_ds,label='ds')
plt.plot(t,a_ds2,label='ds w/datetime (same)')

import statsmodels.formula.api as smf
mod = smf.ols(formula='a~t',data=pd.DataFrame({'a':a,'t':t}))
res = mod.fit()
plt.plot(t[np.isfinite(a)],res.fittedvalues,label='fit')
plt.legend(loc='best')
print(res.summary())
print(res.params)
print(res.bse)
print(smf.ols(formula='a~t',data=pd.DataFrame({'a':a_ds,'t':t})).fit().summary())

#import statsmodels.api as sm
#dta = sm.datasets.co2.load_pandas().data
## deal with missing values. see issue
#dta.co2.interpolate(inplace=True)
#res = sm.tsa.seasonal_decompose(dta.co2)
#plt.subplot(2,1,2)
#plt.plot(dta.index,dta.co2,label='obs')
#plt.plot(dta.index,res.trend,label='trend')
#plt.plot(dta.index,res.trend+res.resid,label='trend+resid')
#plt.plot(dta.index,ds(dta.co2.values,dta.index),label='myds')
#plt.legend(loc='best')


