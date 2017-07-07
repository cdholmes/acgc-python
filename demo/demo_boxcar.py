# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:08:47 2015

@author: cdholmes
"""

import numpy as np
from boxcar import boxcar

# Create a synthetic dataset consisting of a sine wave with noise
x = np.arange(0,4*np.pi,0.1)
y = np.sin(x) + 0.1*np.random.normal(size=len(x))

# Calculate boxcar smoothing
z = boxcar(y, 10)

import matplotlib.pyplot as p
p.clf()
p.plot(x,y)
p.plot(x,z)
