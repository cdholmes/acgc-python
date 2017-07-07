# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:12:55 2016

@author: cdholmes
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


N = 1000
a = 10 # intercept
b = 5  # slope
c = 200 # offset for group 2
n = 200 # magnitude of noise

x = np.arange(N)
y = a + b*x + n*np.random.randn(N)

#model  = smf.ols('y ~ 1 + x',data=dict(x=x,y=y))
#result = model.fit()
#print(result.summary())
#print(result.predict(dict(x=1)))


#df = pd.DataFrame({'x':x,'y':y})
#model2  = smf.ols('y ~ 1 + x',data=df)
#result2 = model2.fit()
#print(result2.predict(dict(x=1)))

group = np.mod(x,3)
y = y + c * group
df = pd.DataFrame({'x':x,'y':y,'group':group})
model3 = smf.ols('y ~ 1 + x + C(group) + x:C(group)',data=df)
result3 = model3.fit()

print(result3.summary())

#print(result3.predict(df))

