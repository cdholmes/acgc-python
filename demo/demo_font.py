#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:38:21 2019

@author: cdholmes
"""

import numpy as np
import matplotlib.pyplot as plt

# Choose the sans-serif font family and make Arial the primary sans-serif font. 
plt.rcParams["font.family"]     = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"

# Create a time series with noise
t = np.sort(np.random.uniform(low = 0, high = 2, size=1000))
ytrue = np.sin( t * 2*np.pi )
y = ytrue + np.random.normal(size=len(t))

# Plot the data
plt.clf()
plt.plot( t, y,     '.', label='obs')
plt.plot( t, ytrue,      label='true')


plt.show()
