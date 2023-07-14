# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:37:08 2015

@author: cdholmes
"""

import numpy as np
import matplotlib.pyplot as plt
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

print('***OLS LINEAR REGRESSION***')
# Create regression model
model_linear = smf.ols('y ~ 1 + t',data)

# Do the regression fit
result_linear = model_linear.fit()

print('\nBest fit parameter values')
print(result_linear.params)


print('***OLS LINEAR REGRESSION, forcing zero intercept***')
# Create regression model
model_linear0 = smf.ols('y ~ 0 + t',data)

# Do the regression fit
result_linear0 = model_linear0.fit()

print('\nBest fit parameter values')
print(result_linear0.params)

print('***OLS REGRESSION***')
# Create regression model
model_ols = smf.ols('y ~ 1 + t + np.square(t)',data)

# Do the regression fit
result_ols = model_ols.fit()

print('\nBest fit parameter values')
print(result_ols.params)

print('\n68% Confidence interval (1-sigma) for fit parameters')
print(result_ols.conf_int(0.68))

# Covariance of fit parameters
pcov = result_ols.cov_params()

# inverse standard deviation of fit parameters
pcor = np.diag(1/np.sqrt(np.diag(pcov)))

print('\n*Covariance* of fit parameters')
print(result_ols.cov_params())

print('\n*Cross correlation* of fit parameters')
print(result_ols.cov_params(pcor))

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
plt.clf()
plt.plot(t,y,'o',label='data')
plt.plot(t,model_linear.predict(result_linear.params), label='Linear')
plt.plot(t,model_linear0.predict(result_linear0.params), label='Linear, zero intercept')
plt.plot(t,model_ols.predict(result_ols.params), label='OLS')
plt.plot(t,model_qr.predict(result_qr.params),label='QuantReg')
plt.plot(t,model_rlm.predict(result_rlm.params), label='RLM')
plt.legend(loc='lower left')
plt.show()
