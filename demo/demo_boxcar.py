# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:08:47 2015

@author: cdholmes
"""

import numpy as np
import acgc.stats as acstats

# Create a synthetic dataset consisting of a sine wave with noise
x = np.arange(0,4*np.pi,0.1)
y = np.sin(x) + 0.1*np.random.normal(size=len(x))

# Calculate boxcar smoothing
z = acstats.boxcar(y, 10)

import matplotlib.pyplot as plt
plt.clf()
plt.plot(x,y,label='noisy')
plt.plot(x,z,label='smoothed')
plt.legend()
plt.show()
