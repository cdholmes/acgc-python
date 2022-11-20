#!/usr/bin/env python3
'''Demonstrate how to use a sqrt axis on a plot'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.scale import FuncScale

n = 1000
x = np.random.lognormal(size=n)
y = np.random.lognormal(size=n)

ax = plt.subplot(1,1,1)
s = ax.scatter(x,y)
m = 2
scalefunc = (lambda x: x**(1/m), lambda x:x**m)
ax.set_yscale('function',functions=scalefunc)
ax.set_xscale('function',functions=scalefunc)