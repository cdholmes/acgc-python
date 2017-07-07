# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:37:08 2015

@author: cdholmes
"""

import numpy as np
import matplotlib.pyplot as p
#from scipy import stats
import pandas as pandas
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Function to normalize data
def norm(v):
    return (v-np.mean(v))/np.std(v)

# Create artificial quadratic data set with error
n=30
t=np.arange(n,dtype='float')
e=np.random.randn(len(t))*30
y=20*t-t**2+e
# Make the first point an outlier
y[0] = y[0] + 400

# Optional: normalize varaibles
#y = norm(y)
#t = norm(t) 

# Convert to data fram
data = pandas.DataFrame({'t':t,'y':y})


print('***OLS REGRESSION***')
# Create regression model
model = smf.ols('y ~ 1 + t + np.square(t)',data)

# Do the regression fit
result = model.fit()

print('\nBest fit parameter values')
print(result.params)

print('\n68% Confidence interval (1-sigma) for fit parameters')
print(result.conf_int(0.68))

# Covariance of fit parameters
pcov = result.cov_params()

# inverse standard deviation of fit parameters
pcor = np.diag(1/np.sqrt(np.diag(pcov)))

print('\n*Covariance* of fit parameters')
print(result.cov_params())

print('\n*Cross correlation* of fit parameters')
print(result.cov_params(pcor))

print('***QUANTILE REGRESSION***')
# Create the model
model_qr = smf.quantreg('y ~ 1 + t + np.square(t)',data)

# Do the fit
result_qr = model_qr.fit()

print('\nBest fit parameter values')
print(result_qr.params)

print('\n68% Confidence interval (1-sigma) for fit parameters')
print(result_qr.conf_int(0.68))

# Covariance of fit parameters
pcov = result_qr.cov_params()

# inverse standard deviation of fit parameters
pcor = np.diag(1/np.sqrt(np.diag(pcov)))

print('\n*Covariance* of fit parameters')
print(result_qr.cov_params())

print('\n*Cross correlation* of fit parameters')
print(result_qr.cov_params(pcor))

print('***RLM REGRESSION***')
# Create regression model
model_rlm = smf.rlm('y ~ 1 + t + np.square(t)',data, M=sm.robust.norms.TukeyBiweight())

# Do the regression fit
result_rlm = model_rlm.fit()

# Display data and best fit
p.clf()
p.plot(t,y,'o',label='data')
p.plot(t,model.predict(result.params), label='OLS')
p.plot(t,model_qr.predict(result_qr.params),label='QuantReg')
p.plot(t,model_rlm.predict(result_rlm.params), label='RLM')
p.legend(loc='lower left')
