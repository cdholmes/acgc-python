#!/usr/bin/env ipython3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

x  = np.arange(5)
y  = np.array([0.5,0.9,1.7,3.3,3.8])
ye = np.array([0.1,0.3,0.1,0.3,0.05])

df = pd.DataFrame({'x':x,'y':y,'ye':ye})

X = sm.add_constant(x)

# OLS
fitols = smf.ols('y ~ 1 + x',data=df).fit()

# Analytical
b   = np.linalg.inv(X.T @ X) @ X.T @ y
bse = np.var(fitols.resid) * np.linalg.inv(X.T @ X )

# WLS
fitwls = smf.wls('y ~ 1 + x',data=df,weights=1/df.ye**2).fit()

# Bootstrap
n=1000
coef = np.zeros((2,n))
for i in range(n):
    # Add gaussian noise
    y2 = y + ye*np.random.randn(5)
    # fit the new data
    fit = sm.OLS(y2,X).fit()
    coef[:,i] = fit.params
    

plt.clf()
plt.scatter(x,y,label='data')
plt.plot(x,fitols.fittedvalues,label='OLS')
plt.plot(x,fitwls.fittedvalues,label='WLS')
plt.legend()
plt.show()

## 
print('\nOLS')
print(fitols.bse)
print('\nOLS + Bootstrap')
print(np.std(coef,axis=1))
print('\nWLS')
print(fitwls.bse)