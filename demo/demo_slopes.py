# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:57:32 2015



@author: cdholmes
"""

import numpy as np
import scipy.stats
from sen_slope import sen_slope 
import matplotlib.pyplot as p


x = np.random.randn(600)
y = x + np.random.randn(600)*0.01

out = scipy.stats.mstats.theilslopes(y,x)

b, bs = sen_slope(x,y)
print(b,np.std(bs)/np.sqrt(len(bs)*2))
print(out[0],(out[3]-out[2])/2)

p.clf()
p.plot(x,y,'o')