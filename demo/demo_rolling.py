# -*- coding: utf-8 -*-
"""
Created on Thu Apr 04 2019

@author: cdholmes
"""


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Create a time series with noise
t = np.sort(np.random.uniform(low = 0, high = 2, size=1000))
ytrue = np.sin( t * 2*np.pi )
y = ytrue + np.random.normal(size=len(t)) #stats.cauchy.rvs(size=len(t))

# Convert t (days) into a datetime array
time = pd.Timestamp('2010-01-01') + pd.TimedeltaIndex( t, 'D' )

# Set up dataframe
data = pd.DataFrame({'y': y, 'ytrue': ytrue}, index=time )

# Apply rolling mean
d2 = data.rolling( '12h' ).mean()

plt.clf()
plt.plot( data.index, data.y,     '.', label='obs')
plt.plot( data.index, data.ytrue,      label='true')
plt.plot( d2.index,   d2.y,       'C3',  label='smooth')


plt.show()




